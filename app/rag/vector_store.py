"""
Vector Store — CRUD với ChromaDB.

ChromaDB lưu vector + document, hỗ trợ tìm kiếm theo cosine similarity.
Giống @Repository trong Spring Boot — tầng thao tác data.
"""

import chromadb
from loguru import logger

from app.core.config import settings

# ChromaDB client (singleton)
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

# Tên collection trong ChromaDB (giống tên bảng trong PostgreSQL)
_COLLECTION_NAME = "talex_videos"


def init_vector_store() -> None:
    """
    Khởi tạo ChromaDB client và collection.
    Gọi 1 lần khi app khởi động.
    """
    global _client, _collection

    logger.info(f"Initializing ChromaDB at: {settings.CHROMA_PERSIST_DIR}")

    # PersistentClient = lưu data xuống disk, restart app vẫn còn
    _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    # get_or_create = lấy collection nếu đã có, tạo mới nếu chưa
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # dùng cosine similarity
    )

    count = _collection.count()
    logger.info(f"ChromaDB ready. Collection '{_COLLECTION_NAME}' has {count} videos.")


def get_collection() -> chromadb.Collection:
    """Lấy collection đã khởi tạo."""
    if _collection is None:
        raise RuntimeError("ChromaDB chưa được khởi tạo. Gọi init_vector_store() trước.")
    return _collection


def add_video(video_id: int, document: str, embedding: list[float], metadata: dict | None = None) -> None:
    """
    Thêm 1 video vào ChromaDB.

    Args:
        video_id: ID video từ PostgreSQL.
        document: Text gốc (title + description + tags).
        embedding: Vector đã chuyển từ embedding model.
        metadata: Thông tin phụ (tags, genre...) — optional.
    """
    collection = get_collection()

    # ChromaDB dùng string ID
    str_id = str(video_id)

    collection.upsert(
        ids=[str_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata] if metadata else None,
    )
    logger.debug(f"Upserted video {video_id} into ChromaDB.")


def delete_video(video_id: int) -> None:
    """Xóa 1 video khỏi ChromaDB."""
    collection = get_collection()
    collection.delete(ids=[str(video_id)])
    logger.debug(f"Deleted video {video_id} from ChromaDB.")


def search_similar(query_embedding: list[float], top_k: int = 10) -> dict:
    """
    Tìm top_k video có vector gần nhất với query.

    Returns:
        {
            "ids": ["1", "3", "50"],
            "distances": [0.05, 0.12, 0.25],
            "documents": ["text gốc video 1", ...]
        }
    """
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
    }


def get_video_count() -> int:
    """Đếm số video trong ChromaDB — dùng cho health check."""
    if _collection is None:
        return 0
    return _collection.count()


def is_connected() -> bool:
    """Kiểm tra ChromaDB đã khởi tạo chưa — dùng cho health check."""
    return _collection is not None
