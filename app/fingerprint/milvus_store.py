"""
Milvus Store — CRUD vectors trong Milvus.

Giống vector_store.py (ChromaDB) nhưng cho fingerprint video/ảnh.
ChromaDB: text search (nhẹ, nhúng trong app).
Milvus: fingerprint search (mạnh, chạy container riêng).
"""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from loguru import logger

from app.core.config import settings
from app.fingerprint.hasher import VECTOR_DIM

# v2 -> v3: media_id/creator_id trước đây KHÔNG có index — mọi delete_by_media_id()
# (expr media_id==X) và search_similar() (expr creator_id!=X, thêm ở bản loại trừ cùng
# creator) đều phải quét toàn bộ scalar data không index, càng chậm dần khi fingerprint
# tích lũy theo thời gian.
#
# v3 -> v4: Milvus không hỗ trợ thêm scalar field mới vào collection đã tồn tại — phải tạo
# collection mới. Thêm 4 field cho "sổ đăng ký chủ sở hữu nội dung gốc" (content-ownership
# registry): content_cluster_id/original_creator_id/first_seen_at/is_violation. Đây là field
# cấp NỘI DUNG (content cluster), không phải cấp media_id — cố tình tách rời khỏi vòng đời
# media_id (mỗi lần re-upload đều delete-by-media_id rồi insert lại) để creator gốc không
# mất quyền sở hữu chỉ vì re-upload. Dữ liệu `_v3` cũ được migrate qua script backfill riêng
# (scripts/backfill_fingerprint_registry.py), không bỏ.
_COLLECTION_NAME = "talex_fingerprints_v4"
_collection: Collection | None = None
_connected: bool = False

# Timeout job-level (asyncio.wait_for) ở kafka_consumer_service.py chỉ hủy việc CHỜ, không
# ép dừng được thread thật đang chạy lời gọi Milvus bên dưới (giống trường hợp boto3, xem
# s3_client.py) — nếu Milvus tự nhiên treo (mạng chập chờn giữa app và container Milvus),
# thread đó có thể chạy ngầm vô thời hạn, dồn lại chiếm hết thread pool dùng chung. Set
# timeout ngắn ở đây để lời gọi tự bỏ cuộc nhanh hơn.
_MILVUS_CALL_TIMEOUT_SECONDS = 30


def init_milvus() -> None:
    """Kết nối Milvus + tạo collection nếu chưa có."""
    global _collection, _connected

    try:
        logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=_MILVUS_CALL_TIMEOUT_SECONDS,
        )

        # Tạo collection nếu chưa có
        if not utility.has_collection(_COLLECTION_NAME):
            _create_collection()
        else:
            _collection = Collection(_COLLECTION_NAME)
            _collection.load()

        _connected = True
        count = _collection.num_entities
        logger.info(f"Milvus ready. Collection '{_COLLECTION_NAME}' has {count} vectors.")

    except Exception as e:
        _connected = False
        logger.warning(f"Milvus connection failed: {e}. Fingerprint features will be unavailable.")


def _create_collection() -> None:
    """Tạo collection schema cho fingerprint."""
    global _collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=50),
        # Lưu kèm creator_id để loại trừ so khớp trong cùng creator (không tự báo "vi
        # phạm" nội dung của chính mình — vd. nhân vật lặp lại xuyên suốt 1 bộ truyện).
        FieldSchema(name="creator_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="timestamp", dtype=DataType.FLOAT),
        # BINARY_VECTOR: pHash là 64 bit nhị phân, phải so bằng Hamming distance —
        # KHÔNG dùng FLOAT_VECTOR + COSINE (sai chuẩn, xem ghi chú ở _COLLECTION_NAME).
        FieldSchema(name="vector", dtype=DataType.BINARY_VECTOR, dim=VECTOR_DIM),
        # --- Content-ownership registry (v4) — cấp NỘI DUNG, không phải cấp media_id ---
        # ID ổn định cho "1 nội dung hình ảnh/video cụ thể", gán 1 lần duy nhất khi nội dung
        # đó lần đầu xuất hiện trong hệ thống, không đổi qua các lần re-upload/upsert khác nhau.
        FieldSchema(name="content_cluster_id", dtype=DataType.VARCHAR, max_length=64),
        # Creator đầu tiên đưa nội dung này vào hệ thống — chủ sở hữu cụm, dùng để luôn
        # cho phép chủ gốc re-upload mà không bị báo vi phạm ngược, bất kể ai khác đã
        # đăng bản sao của nội dung này sau đó.
        FieldSchema(name="original_creator_id", dtype=DataType.VARCHAR, max_length=50),
        # Epoch giây — thời điểm cụm nội dung này được ghi nhận lần đầu. Khác với field
        # "timestamp" ở trên (mốc thời gian TRONG video, không phải thời điểm upload).
        FieldSchema(name="first_seen_at", dtype=DataType.FLOAT),
        # True nếu dòng này là bản sao bị gắn cờ vi phạm (creator khác != original_creator_id
        # của cụm). Vẫn giữ lại để làm bằng chứng/audit, nhưng bị loại khi tra cứu chủ sở hữu
        # để tránh 1 chuỗi bản-sao-của-bản-sao làm sai lệch việc quy về đúng chủ gốc.
        FieldSchema(name="is_violation", dtype=DataType.BOOL),
    ]

    schema = CollectionSchema(fields=fields, description="TaleX video fingerprints (Hamming/pHash)")
    _collection = Collection(name=_COLLECTION_NAME, schema=schema)

    # Index nhị phân + Hamming distance — đúng chuẩn so sánh pHash.
    index_params = {
        "metric_type": "HAMMING",
        "index_type": "BIN_IVF_FLAT",
        "params": {"nlist": 128},
    }
    _collection.create_index(field_name="vector", index_params=index_params)

    # Index scalar cho media_id/creator_id — không có 2 index này, mỗi lần delete/search
    # dùng expr lọc theo 2 field này phải quét toàn bộ dữ liệu không index, chậm dần khi
    # số fingerprint tích lũy tăng lên. INVERTED phù hợp cho lọc khớp chính xác (==, !=).
    _collection.create_index(field_name="media_id", index_params={"index_type": "INVERTED"})
    _collection.create_index(field_name="creator_id", index_params={"index_type": "INVERTED"})
    # content_cluster_id: tra cứu chủ sở hữu cụm nội dung (get_cluster_owner). is_violation:
    # lọc bỏ bản sao đã bị gắn cờ khi tra chủ sở hữu, tránh chuỗi bản-sao-của-bản-sao.
    _collection.create_index(field_name="content_cluster_id", index_params={"index_type": "INVERTED"})
    _collection.create_index(field_name="is_violation", index_params={"index_type": "INVERTED"})
    _collection.load()

    logger.info(
        f"Created Milvus collection '{_COLLECTION_NAME}' with BIN_IVF_FLAT/HAMMING + "
        f"scalar indexes (media_id, creator_id, content_cluster_id, is_violation)."
    )


def get_collection() -> Collection:
    """Lấy collection đã khởi tạo."""
    if _collection is None:
        raise RuntimeError("Milvus chưa được khởi tạo.")
    return _collection


def insert_fingerprints(
    media_id: str,
    creator_id: str,
    fingerprints: list[dict],
    content_cluster_id: str,
    original_creator_id: str,
    first_seen_at: float,
    is_violation: bool,
) -> int:
    """
    Lưu fingerprints của 1 video/ảnh vào Milvus.

    Args:
        media_id: ID video từ Spring Boot.
        creator_id: ID creator sở hữu media này (dùng để loại trừ khi search sau này).
        fingerprints: List of { "timestamp": float, "vector": bytes }
        content_cluster_id: ID cụm nội dung (xem resolve_content_cluster trong content_ownership.py)
            — 1 giá trị duy nhất broadcast cho mọi vector của lần insert này, giống cách
            creator_id đã broadcast hiện tại (không phải giá trị theo từng vector).
        original_creator_id: chủ sở hữu gốc của cụm nội dung này.
        first_seen_at: epoch giây cụm nội dung được ghi nhận lần đầu (giữ nguyên qua các
            lần re-upload của chủ gốc — không phải giờ hiện tại mỗi lần gọi hàm này).
        is_violation: True nếu lần insert này là bản sao bị gắn cờ vi phạm (creator hiện tại
            khác original_creator_id và có match); False cho chủ gốc hoặc nội dung sạch.

    Returns:
        Số vectors đã insert.
    """
    collection = get_collection()

    n = len(fingerprints)
    media_ids = [media_id] * n
    creator_ids = [creator_id] * n
    timestamps = [fp["timestamp"] for fp in fingerprints]
    vectors = [fp["vector"] for fp in fingerprints]
    cluster_ids = [content_cluster_id] * n
    original_creator_ids = [original_creator_id] * n
    first_seen_ats = [first_seen_at] * n
    is_violations = [is_violation] * n

    # KHÔNG gọi flush() ở đây — Milvus search() đã đọc được dữ liệu vừa insert ngay
    # (qua "growing segment" trong RAM) mà không cần đợi flush. flush() ép ghi đĩa
    # sớm, tốn thời gian và bị serialize nội bộ khi nhiều luồng gọi cùng lúc — gây
    # nghẽn tăng dần khi push nhiều ảnh song song (verified: Milvus docs khuyến cáo
    # không flush() sau mỗi insert, để mặc định tự flush theo chu kỳ nền).
    # Thứ tự cột PHẢI khớp đúng thứ tự FieldSchema khai báo ở _create_collection() (trừ
    # "id" auto_id) — Milvus insert theo vị trí, không theo tên.
    collection.insert(
        [
            media_ids,
            creator_ids,
            timestamps,
            vectors,
            cluster_ids,
            original_creator_ids,
            first_seen_ats,
            is_violations,
        ],
        timeout=_MILVUS_CALL_TIMEOUT_SECONDS,
    )

    logger.debug(
        f"Inserted {n} vectors for media_id={media_id}, cluster={content_cluster_id}, "
        f"original_creator={original_creator_id}, is_violation={is_violation}"
    )
    return n


def search_similar(vectors: list[bytes], top_k: int = 5, exclude_creator_id: str | None = None) -> list[dict]:
    """
    Tìm vectors giống nhất trong Milvus (Hamming distance trên BINARY_VECTOR).

    Args:
        vectors: List of query vectors (bytes đã packbits, xem hasher.py).
        top_k: Số kết quả mỗi vector.
        exclude_creator_id: Loại trừ NGAY TẠI MILVUS (không phải lọc sau ở matcher.py) —
            nếu 1 creator có nhiều trang giống nhau (vd nhân vật lặp lại xuyên truyện), các
            trang CŨ của chính họ dễ chiếm hết top_k vì giống nhau rất cao, khiến 1 vi phạm
            thật với creator KHÁC (đứng hạng thấp hơn) không bao giờ lọt vào top_k để được
            xét tới — lọc ở tầng Milvus đảm bảo top_k luôn là kết quả từ người khác.

    Returns:
        List of { "query_index": int, "media_id": int, "creator_id": str,
                   "timestamp": float, "score": float, "content_cluster_id": str | None,
                   "original_creator_id": str | None, "first_seen_at": float | None,
                   "is_violation": bool }
        "score" đã quy đổi về thang 0..1 (càng cao càng giống — giữ đúng contract cũ
        cho matcher.py/FINGERPRINT_SIMILARITY_THRESHOLD), dù Milvus trả Hamming distance
        thô (số nguyên, thấp = giống).
    """
    collection = get_collection()

    search_params = {"metric_type": "HAMMING", "params": {"nprobe": 16}}
    # creator_id luôn là UUID nội bộ (không phải input người dùng tự do), nhưng vẫn escape
    # dấu " cho chắc — tránh làm hỏng cú pháp expr nếu giá trị bất thường lọt vào.
    expr = f'creator_id != "{exclude_creator_id.replace(chr(34), "")}"' if exclude_creator_id else None

    results = collection.search(
        data=vectors,
        anns_field="vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "media_id",
            "creator_id",
            "timestamp",
            "content_cluster_id",
            "original_creator_id",
            "first_seen_at",
            "is_violation",
        ],
        timeout=_MILVUS_CALL_TIMEOUT_SECONDS,
    )

    matches = []
    for query_idx, hits in enumerate(results):
        for hit in hits:
            hamming_distance = hit.score
            similarity = 1.0 - (hamming_distance / VECTOR_DIM)
            matches.append({
                "query_index": query_idx,
                "media_id": hit.entity.get("media_id"),
                "creator_id": hit.entity.get("creator_id"),
                "timestamp": hit.entity.get("timestamp"),
                "score": similarity,
                # .get() với default an toàn — phòng hờ dòng dữ liệu cũ trước migrate
                # chưa có các field registry này (không nên xảy ra sau khi backfill xong,
                # nhưng an toàn hơn để không crash toàn bộ search nếu có sót).
                "content_cluster_id": hit.entity.get("content_cluster_id"),
                "original_creator_id": hit.entity.get("original_creator_id"),
                "first_seen_at": hit.entity.get("first_seen_at"),
                "is_violation": bool(hit.entity.get("is_violation")),
            })

    return matches


def query_cluster_rows(content_cluster_id: str, only_non_violation: bool = True) -> list[dict]:
    """
    Đọc thẳng các dòng trong Milvus thuộc 1 content_cluster_id — CRUD thuần, không có logic
    "ai là chủ sở hữu" (xem app/fingerprint/content_ownership.py:get_cluster_owner cho việc đó).

    Args:
        content_cluster_id: ID cụm nội dung cần đọc.
        only_non_violation: True (mặc định) → chỉ lấy dòng is_violation==false (dòng "sạch",
            có thể là chủ sở hữu). False → lấy tất cả dòng kể cả bản sao bị gắn cờ.

    Returns:
        List of { "original_creator_id": str, "first_seen_at": float }.
    """
    collection = get_collection()
    # content_cluster_id do hệ thống tự sinh (uuid4), nhưng vẫn escape dấu " cho nhất quán
    # với các expr khác trong module này (search_similar, delete_by_media_id).
    escaped = content_cluster_id.replace(chr(34), "")
    expr = f'content_cluster_id == "{escaped}"'
    if only_non_violation:
        expr += " && is_violation == false"
    return collection.query(
        expr=expr,
        output_fields=["original_creator_id", "first_seen_at"],
        timeout=_MILVUS_CALL_TIMEOUT_SECONDS,
    )


def delete_by_media_id(media_id: str) -> int:
    """Xóa tất cả vectors của 1 video."""
    collection = get_collection()

    # media_id tới từ Kafka (content-media-delete đọc thẳng dict, không qua Pydantic
    # validate) — escape dấu " giống search_similar() ở trên, tránh 1 giá trị bất thường
    # làm hỏng/mở rộng cú pháp boolean expr (có thể xóa nhầm fingerprint của media khác).
    expr = f'media_id == "{media_id.replace(chr(34), "")}"'
    result = collection.delete(expr, timeout=_MILVUS_CALL_TIMEOUT_SECONDS)
    # Tương tự insert_fingerprints() — search() vẫn thấy đúng trạng thái đã xóa mà
    # không cần flush() ngay, tránh nghẽn khi nhiều luồng cùng gọi song song.

    count = result.delete_count
    logger.debug(f"Deleted {count} vectors for media_id={media_id}")
    return count


def get_count() -> int:
    """Đếm tổng vectors trong collection."""
    if _collection is None:
        return 0
    _collection.flush()
    return _collection.num_entities


def is_connected() -> bool:
    """Kiểm tra Milvus đã kết nối chưa."""
    return _connected
