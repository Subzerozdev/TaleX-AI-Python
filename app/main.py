"""
Điểm khởi chạy của TaleX AI Service.
Giống @SpringBootApplication trong Spring Boot.

Chạy: uvicorn app.main:app --reload
Swagger UI: http://localhost:8000/docs
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.error_handler import register_error_handlers
from app.core.logging_config import setup_logging
from app.core.rate_limiter import limiter
from app.fingerprint import milvus_store
from app.kafka.kafka_consumer_service import consume_loop
from app.kafka.kafka_producer_service import start_producer, stop_producer
from app.llm import gemini_client
from app.rag import embeddings, vector_store
from app.routers import chat, content, fingerprint, health, moderation, search, sync


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Startup/Shutdown logic.
    Giống @PostConstruct trong Spring Boot — chạy khi app khởi động.
    """
    # === STARTUP ===
    setup_logging()
    logger.info("Starting TaleX AI Service...")

    # 1. Load embedding model
    embeddings.load_model()

    # 2. Khởi tạo ChromaDB
    vector_store.init_vector_store()

    # 3. Khởi tạo Gemini client
    gemini_client.init_gemini()

    # 4. Khởi tạo Milvus (Content ID)
    milvus_store.init_milvus()

    # 5. Seed data nếu ChromaDB rỗng
    if vector_store.get_video_count() == 0:
        _seed_data()

    # 6. Kafka producer + consumer (content pipeline)
    await start_producer()
    consumer_task = asyncio.create_task(consume_loop())

    logger.info("TaleX AI Service ready!")

    yield  # app chạy ở đây

    # === SHUTDOWN ===
    logger.info("Shutting down TaleX AI Service...")
    consumer_task.cancel()
    await stop_producer()


def _seed_data():
    """Load video mẫu từ file JSON vào ChromaDB."""
    seed_file = Path("data/seed_videos.json")

    if not seed_file.exists():
        logger.warning(f"Seed file not found: {seed_file}")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        videos = json.load(f)

    logger.info(f"Seeding {len(videos)} videos into ChromaDB...")

    # Ghép text từ title + description + tags
    texts = []
    for video in videos:
        text = f"{video['title']}. {video['description']}. Tags: {', '.join(video['tags'])}"
        texts.append(text)

    # Chuyển tất cả text → vectors cùng lúc (nhanh hơn từng cái)
    vectors = embeddings.embed_texts(texts)

    # Lưu vào ChromaDB
    for video, text, vector in zip(videos, texts, vectors):
        vector_store.add_video(
            video_id=video["video_id"],
            document=text,
            embedding=vector,
            metadata={"tags": ",".join(video["tags"])},
        )

    logger.info(f"Seeded {len(videos)} videos successfully.")


# Tạo FastAPI app
app = FastAPI(
    title="TaleX AI Service",
    description="AI Service cho nền tảng TaleX — chatbot, search, tagging, moderation.",
    version="0.1.0",
    lifespan=lifespan,
)

# Đăng ký rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Đăng ký global error handlers
register_error_handlers(app)

# Đăng ký routers (giống Spring Boot scan @RestController)
app.include_router(health.router, tags=["Health"])
app.include_router(search.router)
app.include_router(sync.router)
app.include_router(chat.router)
app.include_router(content.router)
app.include_router(moderation.router)
app.include_router(fingerprint.router)
