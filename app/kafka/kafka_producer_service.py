"""Kafka producer — send results back to Spring Boot."""

import json
import ssl

from aiokafka import AIOKafkaProducer
from loguru import logger

from app.core.config import settings
from app.kafka.kafka_config import TOPIC_COPYRIGHT_RESULT, TOPIC_MODERATION_RESULT

_producer: AIOKafkaProducer | None = None


def _build_ssl_context() -> ssl.SSLContext | None:
    """Build SSL context from Aiven PEM cert paths."""
    if not settings.KAFKA_SSL_CAFILE:
        return None
    ctx = ssl.create_default_context(cafile=settings.KAFKA_SSL_CAFILE)
    ctx.load_cert_chain(
        certfile=settings.KAFKA_SSL_CERTFILE,
        keyfile=settings.KAFKA_SSL_KEYFILE,
    )
    return ctx


async def start_producer():
    """Start Kafka producer. No-op if KAFKA_BOOTSTRAP_SERVERS not set."""
    global _producer
    if not settings.KAFKA_BOOTSTRAP_SERVERS:
        logger.warning("KAFKA_BOOTSTRAP_SERVERS not set — Kafka producer disabled")
        return

    ssl_ctx = _build_ssl_context()
    kwargs = {
        "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
        "key_serializer": lambda k: k.encode("utf-8") if k else None,
    }
    if ssl_ctx:
        kwargs["security_protocol"] = "SSL"
        kwargs["ssl_context"] = ssl_ctx

    _producer = AIOKafkaProducer(**kwargs)
    await _producer.start()
    logger.info("Kafka producer started")


async def stop_producer():
    """Graceful shutdown."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")


async def send_copyright_result(media_id: str, result: dict):
    """Send copyright check result to Spring Boot."""
    if _producer is None:
        logger.warning("Kafka producer not available, skipping send")
        return
    await _producer.send_and_wait(TOPIC_COPYRIGHT_RESULT, key=media_id, value=result)
    logger.info(f"Copyright result sent: mediaId={media_id}")


async def send_moderation_result(media_id: str, result: dict):
    """Send moderation result to Spring Boot."""
    if _producer is None:
        logger.warning("Kafka producer not available, skipping send")
        return
    await _producer.send_and_wait(TOPIC_MODERATION_RESULT, key=media_id, value=result)
    logger.info(f"Moderation result sent: mediaId={media_id}")
