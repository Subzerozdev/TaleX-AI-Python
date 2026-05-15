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


# Singleton — toàn app dùng chung 1 instance
settings = Settings()
