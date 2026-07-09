"""
Milvus Store — CRUD vectors cho Recommendation System (Series).
"""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
)
from loguru import logger

_COLLECTION_NAME = "talex_series_recommendations"
_collection: Collection | None = None

# all-MiniLM-L6-v2 embeddings dim is 384
VECTOR_DIM = 384

def init_recommendation_milvus() -> None:
    """Khởi tạo collection cho recommendation."""
    global _collection

    try:
        if not utility.has_collection(_COLLECTION_NAME):
            _create_collection()
        else:
            _collection = Collection(_COLLECTION_NAME)
            _collection.load()

        count = _collection.num_entities
        logger.info(f"Milvus Recommendation ready. Collection '{_COLLECTION_NAME}' has {count} vectors.")

    except Exception as e:
        logger.warning(f"Milvus recommendation collection failed: {e}")

def _create_collection() -> None:
    global _collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="series_id", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    ]

    schema = CollectionSchema(fields=fields, description="TaleX Series Recommendations")
    _collection = Collection(name=_COLLECTION_NAME, schema=schema)

    # Tạo index cho vector field (HNSW)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    _collection.create_index(field_name="vector", index_params=index_params)
    _collection.load()

    logger.info(f"Created Milvus collection '{_COLLECTION_NAME}' with HNSW index.")

def get_collection() -> Collection:
    if _collection is None:
        raise RuntimeError("Milvus recommendation chưa được khởi tạo.")
    return _collection

def insert_series_vector(series_id: str, vector: list[float]) -> None:
    """Lưu vector của 1 series vào Milvus."""
    collection = get_collection()
    
    # Xóa cũ nếu có để tránh trùng lặp
    delete_by_series_id(series_id)

    collection.insert([[series_id], [vector]])
    collection.flush()
    logger.debug(f"Inserted vector for series_id={series_id}")

def search_similar_series(vector: list[float], top_k: int = 10) -> list[str]:
    """Tìm top_k series giống nhất."""
    collection = get_collection()

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    results = collection.search(
        data=[vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["series_id"],
    )

    matches = []
    # results[0] contains matches for the first (and only) query vector
    for hit in results[0]:
        matches.append(hit.entity.get("series_id"))

    return matches

def delete_by_series_id(series_id: str) -> None:
    """Xóa vector của 1 series."""
    collection = get_collection()
    expr = f'series_id == "{series_id}"'
    collection.delete(expr)
    collection.flush()
