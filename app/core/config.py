"""
Cấu hình chung cho app.
Đọc biến môi trường từ file .env (giống application.properties trong Spring Boot).
"""

import os
from dotenv import load_dotenv

# Load file .env vào environment variables
load_dotenv()


class Settings:
    """Tập trung mọi config ở 1 chỗ — dễ quản lý, dễ thay đổi."""

    # Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Server
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

    # Rate Limiting
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "10/minute")
    RATE_LIMIT_SEARCH: str = os.getenv("RATE_LIMIT_SEARCH", "30/minute")

    # Timeouts (giây)
    GEMINI_TIMEOUT: int = 10
    CHROMADB_TIMEOUT: int = 5

    # Search
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 50

    # Chat history
    MAX_HISTORY_MESSAGES: int = 20

    # Milvus (Vector DB cho Content ID fingerprint)
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))

    # Fingerprint
    FINGERPRINT_FPS: int = int(os.getenv("FINGERPRINT_FPS", "1"))
    FINGERPRINT_SIMILARITY_THRESHOLD: float = float(os.getenv("FINGERPRINT_SIMILARITY_THRESHOLD", "0.85"))
    FINGERPRINT_MIN_MATCH_SECONDS: int = int(os.getenv("FINGERPRINT_MIN_MATCH_SECONDS", "5"))
    FINGERPRINT_MAX_GAP_SECONDS: int = int(os.getenv("FINGERPRINT_MAX_GAP_SECONDS", "2"))
    FINGERPRINT_MAX_FILE_SIZE_MB: int = int(os.getenv("FINGERPRINT_MAX_FILE_SIZE_MB", "100"))

    # Kafka (Aiven)
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    KAFKA_SSL_CAFILE: str = os.getenv("KAFKA_SSL_CAFILE", "")
    KAFKA_SSL_CERTFILE: str = os.getenv("KAFKA_SSL_CERTFILE", "")
    KAFKA_SSL_KEYFILE: str = os.getenv("KAFKA_SSL_KEYFILE", "")
    KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "python-content-pipeline-group")

    # AWS S3 + Rekognition
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "talex-media-139139347126")

    # Rekognition Content Moderation
    REKOGNITION_CONFIDENCE_THRESHOLD: float = float(os.getenv("REKOGNITION_CONFIDENCE_THRESHOLD", "80.0"))
    REKOGNITION_MAX_FRAMES: int = int(os.getenv("REKOGNITION_MAX_FRAMES", "30"))
    MODERATION_FRAME_INTERVAL: float = float(os.getenv("MODERATION_FRAME_INTERVAL", "2.0"))


# Singleton — toàn app dùng chung 1 instance
settings = Settings()
