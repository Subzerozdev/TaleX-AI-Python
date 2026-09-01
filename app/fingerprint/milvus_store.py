

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


_COLLECTION_NAME = "talex_fingerprints_v4"
_collection: Collection | None = None
_connected: bool = False


_MILVUS_CALL_TIMEOUT_SECONDS = 30


def init_milvus() -> None:
    global _collection, _connected
    try:
        logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            timeout=_MILVUS_CALL_TIMEOUT_SECONDS,
        )
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

    global _collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="media_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="creator_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="timestamp", dtype=DataType.FLOAT),
        FieldSchema(name="vector", dtype=DataType.BINARY_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="content_cluster_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="original_creator_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="first_seen_at", dtype=DataType.FLOAT),
        FieldSchema(name="is_violation", dtype=DataType.BOOL),
    ]

    schema = CollectionSchema(fields=fields, description="TaleX video fingerprints (Hamming/pHash)")
    _collection = Collection(name=_COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "HAMMING",
        "index_type": "BIN_IVF_FLAT",
        "params": {"nlist": 128},
    }
    _collection.create_index(field_name="vector", index_params=index_params)

    _collection.create_index(field_name="media_id", index_params={"index_type": "INVERTED"})
    _collection.create_index(field_name="creator_id", index_params={"index_type": "INVERTED"})
    _collection.create_index(field_name="content_cluster_id", index_params={"index_type": "INVERTED"})
    _collection.create_index(field_name="is_violation", index_params={"index_type": "INVERTED"})
    _collection.load()

    logger.info(
        f"Created Milvus collection '{_COLLECTION_NAME}' with BIN_IVF_FLAT/HAMMING + "
        f"scalar indexes (media_id, creator_id, content_cluster_id, is_violation)."
    )


def get_collection() -> Collection:
    if _collection is None:
        raise RuntimeError("Milvus has not been initialized.")
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
    collection = get_collection()

    search_params = {"metric_type": "HAMMING", "params": {"nprobe": 16}}
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
                "content_cluster_id": hit.entity.get("content_cluster_id"),
                "original_creator_id": hit.entity.get("original_creator_id"),
                "first_seen_at": hit.entity.get("first_seen_at"),
                "is_violation": bool(hit.entity.get("is_violation")),
            })

    return matches


def query_cluster_rows(content_cluster_id: str, only_non_violation: bool = True) -> list[dict]:
    collection = get_collection()
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
    collection = get_collection()

    expr = f'media_id == "{media_id.replace(chr(34), "")}"'
    result = collection.delete(expr, timeout=_MILVUS_CALL_TIMEOUT_SECONDS)
    count = result.delete_count
    logger.debug(f"Deleted {count} vectors for media_id={media_id}")
    return count

def get_count() -> int:
    if _collection is None:
        return 0
    _collection.flush()
    return _collection.num_entities


def is_connected() -> bool:
    return _connected
