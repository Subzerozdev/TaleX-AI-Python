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
# tích lũy theo thời gian. Đổi tên để Milvus tự tạo collection sạch với index scalar mới
# (xem _create_collection) — dữ liệu fingerprint cũ (đang ở giai đoạn test) không cần
# migrate.
_COLLECTION_NAME = "talex_fingerprints_v3"
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
    _collection.load()

    logger.info(f"Created Milvus collection '{_COLLECTION_NAME}' with BIN_IVF_FLAT/HAMMING + scalar indexes.")


def get_collection() -> Collection:
    """Lấy collection đã khởi tạo."""
    if _collection is None:
        raise RuntimeError("Milvus chưa được khởi tạo.")
    return _collection


def insert_fingerprints(media_id: str, creator_id: str, fingerprints: list[dict]) -> int:
    """
    Lưu fingerprints của 1 video/ảnh vào Milvus.

    Args:
        media_id: ID video từ Spring Boot.
        creator_id: ID creator sở hữu media này (dùng để loại trừ khi search sau này).
        fingerprints: List of { "timestamp": float, "vector": bytes }

    Returns:
        Số vectors đã insert.
    """
    collection = get_collection()

    media_ids = [media_id] * len(fingerprints)
    creator_ids = [creator_id] * len(fingerprints)
    timestamps = [fp["timestamp"] for fp in fingerprints]
    vectors = [fp["vector"] for fp in fingerprints]

    # KHÔNG gọi flush() ở đây — Milvus search() đã đọc được dữ liệu vừa insert ngay
    # (qua "growing segment" trong RAM) mà không cần đợi flush. flush() ép ghi đĩa
    # sớm, tốn thời gian và bị serialize nội bộ khi nhiều luồng gọi cùng lúc — gây
    # nghẽn tăng dần khi push nhiều ảnh song song (verified: Milvus docs khuyến cáo
    # không flush() sau mỗi insert, để mặc định tự flush theo chu kỳ nền).
    collection.insert([media_ids, creator_ids, timestamps, vectors], timeout=_MILVUS_CALL_TIMEOUT_SECONDS)

    logger.debug(f"Inserted {len(fingerprints)} vectors for media_id={media_id}")
    return len(fingerprints)


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
                   "timestamp": float, "score": float }
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
        output_fields=["media_id", "creator_id", "timestamp"],
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
            })

    return matches


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
