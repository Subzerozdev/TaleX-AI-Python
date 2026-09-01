

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
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
from app.routers import chat, content, fingerprint, health, moderation, recommendation, search, sync, watermark


@asynccontextmanager
async def lifespan(application: FastAPI):

    # === STARTUP ===
    setup_logging()
    logger.info("Starting TaleX AI Service...")
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=16))
    embeddings.load_model()
    vector_store.init_vector_store()
    gemini_client.init_gemini()
    milvus_store.init_milvus()
    from app.rag.milvus_recommendation_store import init_recommendation_milvus
    init_recommendation_milvus()

    if vector_store.get_video_count() == 0:
        _seed_data()

    from app.db.mongodb import init_mongodb, close_mongodb
    await init_mongodb()

    await start_producer()
    consumer_task = asyncio.create_task(_run_consumer_forever())

    logger.info("TaleX AI Service ready!")

    yield
    logger.info("Shutting down TaleX AI Service...")
    consumer_task.cancel()
    await stop_producer()
    await close_mongodb()


async def _run_consumer_forever():
    while True:
        try:
            await consume_loop()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Kafka consumer loop crashed, restarting in 5s: {e}", exc_info=True)
        await asyncio.sleep(5)


def _seed_data():
    seed_file = Path("data/seed_videos.json")

    if not seed_file.exists():
        logger.warning(f"Seed file not found: {seed_file}")
        return

    with open(seed_file, "r", encoding="utf-8") as f:
        videos = json.load(f)

    logger.info(f"Seeding {len(videos)} videos into ChromaDB...")

    texts = []
    for video in videos:
        text = f"{video['title']}. {video['description']}. Tags: {', '.join(video['tags'])}"
        texts.append(text)

    vectors = embeddings.embed_texts(texts)

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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

register_error_handlers(app)

app.include_router(health.router, tags=["Health"])
app.include_router(search.router)
app.include_router(sync.router)
app.include_router(chat.router)
app.include_router(content.router)
app.include_router(moderation.router)
app.include_router(fingerprint.router)
app.include_router(recommendation.router)
app.include_router(watermark.router)