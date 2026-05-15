"""
Embedding Module — Chuyển text thành vector.

Dùng model all-MiniLM-L6-v2 (chạy local, miễn phí, ~80MB).
Model tự download lần đầu chạy.

Giống: Jackson chuyển JSON → Java Object trong Spring Boot.
Embedding Model chuyển text → vector (dãy 384 con số).
"""

from sentence_transformers import SentenceTransformer
from loguru import logger

# Model name — dùng chung cho cả sync và search
_MODEL_NAME = "all-MiniLM-L6-v2"

# Biến lưu model đã load (singleton)
_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """
    Load embedding model vào RAM.
    Gọi 1 lần khi app khởi động. Các lần sau dùng lại model đã load.
    """
    global _model

    if _model is not None:
        return _model

    logger.info(f"Loading embedding model: {_MODEL_NAME}...")
    _model = SentenceTransformer(_MODEL_NAME)
    logger.info(f"Embedding model loaded. Dimension: {_model.get_embedding_dimension()}")

    return _model


def get_model() -> SentenceTransformer:
    """Lấy model đã load. Raise lỗi nếu chưa load."""
    if _model is None:
        raise RuntimeError("Embedding model chưa được load. Gọi load_model() trước.")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Chuyển 1 đoạn text thành vector.

    Input:  "Hunter yếu nhất bước vào dungeon"
    Output: [0.12, -0.45, 0.78, ..., 0.21]  (384 con số)
    """
    model = get_model()
    # encode() trả numpy array → chuyển thành list Python
    vector = model.encode(text).tolist()
    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Chuyển nhiều đoạn text thành vectors cùng lúc (nhanh hơn gọi từng cái).
    Dùng khi seed data hoặc sync nhiều video.
    """
    model = get_model()
    vectors = model.encode(texts).tolist()
    return vectors


def is_loaded() -> bool:
    """Kiểm tra model đã load chưa — dùng cho health check."""
    return _model is not None
