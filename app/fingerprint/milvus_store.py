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

_COLLECTION_NAME = "talex_fingerprints"
_collection: Collection | None = None
_connected: bool = False


def init_milvus() -> None:
    """Kết nối Milvus + tạo collection nếu chưa có."""
    global _collection, _connected

    try:
        logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
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
        FieldSchema(name="media_id", dtype=DataType.INT64),
        FieldSchema(name="timestamp", dtype=DataType.FLOAT),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]

    schema = CollectionSchema(fields=fields, description="TaleX video fingerprints")
    _collection = Collection(name=_COLLECTION_NAME, schema=schema)

    # Tạo index cho vector field (IVF_FLAT + cosine similarity)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    _collection.create_index(field_name="vector", index_params=index_params)
    _collection.load()

    logger.info(f"Created Milvus collection '{_COLLECTION_NAME}' with IVF_FLAT index.")


def get_collection() -> Collection:
    """Lấy collection đã khởi tạo."""
    if _collection is None:
        raise RuntimeError("Milvus chưa được khởi tạo.")
    return _collection


def insert_fingerprints(media_id: int, fingerprints: list[dict]) -> int:
    """
    Lưu fingerprints của 1 video/ảnh vào Milvus.

    Args:
        media_id: ID video từ Spring Boot.
        fingerprints: List of { "timestamp": float, "vector": list[float] }

    Returns:
        Số vectors đã insert.
    """
    collection = get_collection()

    media_ids = [media_id] * len(fingerprints)
    timestamps = [fp["timestamp"] for fp in fingerprints]
    vectors = [fp["vector"] for fp in fingerprints]

    collection.insert([media_ids, timestamps, vectors])
    collection.flush()

    logger.debug(f"Inserted {len(fingerprints)} vectors for media_id={media_id}")
    return len(fingerprints)


def search_similar(vectors: list[list[float]], top_k: int = 5) -> list[dict]:
    """
    Tìm vectors giống nhất trong Milvus.

    Args:
        vectors: List of query vectors.
        top_k: Số kết quả mỗi vector.

    Returns:
        List of { "query_index": int, "media_id": int, "timestamp": float, "score": float }
    """
    collection = get_collection()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    results = collection.search(
        data=vectors,
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["media_id", "timestamp"],
    )

    matches = []
    for query_idx, hits in enumerate(results):
        for hit in hits:
            matches.append({
                "query_index": query_idx,
                "media_id": hit.entity.get("media_id"),
                "timestamp": hit.entity.get("timestamp"),
                "score": hit.score,
            })

    return matches


def delete_by_media_id(media_id: int) -> int:
    """Xóa tất cả vectors của 1 video."""
    collection = get_collection()

    expr = f"media_id == {media_id}"
    result = collection.delete(expr)
    collection.flush()

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
