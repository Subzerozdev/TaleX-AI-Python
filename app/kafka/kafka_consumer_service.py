"""Kafka consumer — receive pipeline jobs from Spring Boot, process, send results."""

import json
import ssl
from datetime import datetime

from aiokafka import AIOKafkaConsumer
from loguru import logger

from app.aws.s3_client import download_from_s3
from app.core.config import settings
from app.kafka.kafka_config import TOPIC_PIPELINE_JOB, TOPIC_MODERATION_JOB
from app.kafka.kafka_producer_service import send_copyright_result, send_moderation_result
from app.schemas.kafka_messages import PipelineJobMessage
from app.services.fingerprint_service import process_fingerprint
from app.services.video_moderation_service import moderate_media


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


async def consume_loop():
    """Main consumer loop — runs as asyncio background task."""
    if not settings.KAFKA_BOOTSTRAP_SERVERS:
        logger.warning("KAFKA_BOOTSTRAP_SERVERS not set — Kafka consumer disabled")
        return

    ssl_ctx = _build_ssl_context()
    kwargs = {
        "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "group_id": settings.KAFKA_CONSUMER_GROUP,
        "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
        "auto_offset_reset": "earliest",
        "enable_auto_commit": True,
    }
    if ssl_ctx:
        kwargs["security_protocol"] = "SSL"
        kwargs["ssl_context"] = ssl_ctx

    consumer = AIOKafkaConsumer(TOPIC_PIPELINE_JOB, TOPIC_MODERATION_JOB, **kwargs)
    await consumer.start()
    logger.info(f"Kafka consumer started — listening on [{TOPIC_PIPELINE_JOB}, {TOPIC_MODERATION_JOB}]")

    try:
        async for msg in consumer:
            try:
                if msg.topic == TOPIC_PIPELINE_JOB:
                    await _process_pipeline_job(msg.value)
                elif msg.topic == TOPIC_MODERATION_JOB:
                    await _process_moderation_job(msg.value)
            except Exception as e:
                logger.error(f"Job processing failed (topic={msg.topic}): {e}", exc_info=True)
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped")


async def _process_pipeline_job(data: dict):
    """Process a single pipeline job: download from S3, fingerprint, send result."""
    job = PipelineJobMessage.model_validate(data)
    logger.info(f"Processing pipeline job: mediaId={job.media_id}, type={job.media_type}")

    try:
        # Download file from S3
        file_bytes = download_from_s3(job.s3_key, job.s3_bucket)
        filename = job.s3_key.rsplit("/", 1)[-1] if "/" in job.s3_key else job.s3_key

        # Run fingerprint pipeline
        response = process_fingerprint(job.media_id, file_bytes, filename)

        # Build camelCase result for Spring Boot
        result = {
            "mediaId": job.media_id,
            "correlationId": job.correlation_id,
            "contentId": response.content_id,
            "isDuplicate": response.is_duplicate,
            "overallSimilarity": response.overall_similarity,
            "fingerprintCount": response.fingerprint_count,
            "violations": [
                {
                    "sourceMediaId": str(v.source_media_id),
                    "startTimeTarget": v.start_time_target,
                    "endTimeTarget": v.end_time_target,
                    "startTimeSource": v.start_time_source,
                    "endTimeSource": v.end_time_source,
                    "similarityScore": v.similarity_score,
                    "violationType": v.violation_type,
                }
                for v in response.violations
            ],
            "processedAt": datetime.utcnow().isoformat(),
            "success": True,
            "errorMessage": None,
        }
        await send_copyright_result(job.media_id, result)

    except Exception as e:
        logger.error(f"Fingerprint failed for mediaId={job.media_id}: {e}")
        error_result = {
            "mediaId": job.media_id,
            "correlationId": job.correlation_id,
            "contentId": None,
            "isDuplicate": False,
            "overallSimilarity": 0.0,
            "fingerprintCount": 0,
            "violations": [],
            "processedAt": datetime.utcnow().isoformat(),
            "success": False,
            "errorMessage": str(e),
        }
        await send_copyright_result(job.media_id, error_result)


async def _process_moderation_job(data: dict):
    """Process moderation job: download from S3, run Rekognition, send result."""
    job = PipelineJobMessage.model_validate(data)
    logger.info(f"Processing moderation job: mediaId={job.media_id}, type={job.media_type}")

    try:
        file_bytes = download_from_s3(job.s3_key, job.s3_bucket)
        result = moderate_media(file_bytes, job.media_type, job.media_id, job.correlation_id)
        await send_moderation_result(job.media_id, result)
    except Exception as e:
        logger.error(f"Moderation failed for mediaId={job.media_id}: {e}")
        error_result = {
            "mediaId": job.media_id,
            "correlationId": job.correlation_id,
            "isSafe": False,
            "primaryLabel": None,
            "confidenceScore": 0.0,
            "violations": [],
            "rawResponse": "",
            "processedAt": datetime.utcnow().isoformat(),
            "success": False,
            "errorMessage": str(e),
        }
        await send_moderation_result(job.media_id, error_result)
