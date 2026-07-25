"""Kafka consumer — receive pipeline jobs from Spring Boot, process, send results."""

import asyncio
import json
import ssl
from datetime import datetime

from aiokafka import AIOKafkaConsumer
from loguru import logger

from app.aws.s3_client import download_from_s3, upload_to_s3
from app.core.config import settings
from app.kafka.kafka_config import (
    TOPIC_PIPELINE_JOB,
    TOPIC_MODERATION_JOB,
    TOPIC_RECOMMENDATION_SYNC,
    TOPIC_MEDIA_DELETE,
)
from app.kafka.kafka_producer_service import send_copyright_result, send_moderation_result
from app.schemas.kafka_messages import PipelineJobMessage
from app.services.fingerprint_service import process_fingerprint, delete_fingerprint
from app.services.video_moderation_service import moderate_media
from app.services.recommendation_service import process_series_upsert
from app.services.preview_service import generate_image_preview, generate_video_preview

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
        # Commit thủ công sau khi xử lý xong (xem bên dưới) — auto-commit theo thời
        # gian có thể đánh dấu "đã đọc" một job đang xử lý dở, nếu service crash/restart
        # đúng lúc đó thì job bị mất vĩnh viễn, không retry, không báo lỗi cho BE.
        "enable_auto_commit": False,
        "max_poll_interval_ms": 300000,
    }
    if ssl_ctx:
        kwargs["security_protocol"] = "SSL"
        kwargs["ssl_context"] = ssl_ctx

    consumer = AIOKafkaConsumer(
        TOPIC_PIPELINE_JOB, TOPIC_MODERATION_JOB, TOPIC_RECOMMENDATION_SYNC, TOPIC_MEDIA_DELETE, **kwargs
    )
    await consumer.start()
    logger.info(
        f"Kafka consumer started — listening on "
        f"[{TOPIC_PIPELINE_JOB}, {TOPIC_MODERATION_JOB}, {TOPIC_RECOMMENDATION_SYNC}, {TOPIC_MEDIA_DELETE}]"
    )

    # Kết nối TCP tới Aiven Kafka có thể bị NAT/firewall âm thầm cắt khi idle lâu
    # (không có FIN/RST báo về) — nếu không bọc timeout, await getmany()/commit() có
    # thể treo VÔ THỜI HẠN (đã xảy ra thật: treo 7+ tiếng, không lỗi, không crash, không
    # job nào được xử lý tiếp). Bọc timeout cứng để biến "treo im lặng" thành lỗi rõ
    # ràng, cho phép supervisor bên ngoài (main.py) phát hiện và tự kết nối lại.
    NETWORK_CALL_TIMEOUT_SECONDS = 30

    try:
        while True:
            # Lấy 1 batch message thay vì từng cái — cho phép xử lý song song trong
            # cùng 1 consumer, tránh episode nhiều trang (100 job Content ID + 100 job
            # Kiểm duyệt) phải xếp hàng tuần tự dù mỗi job riêng lẻ chỉ mất vài giây.
            batches = await asyncio.wait_for(
                consumer.getmany(timeout_ms=1000, max_records=10),
                timeout=NETWORK_CALL_TIMEOUT_SECONDS,
            )
            if not batches:
                continue

            tasks = [
                _dispatch_job(msg)
                for messages in batches.values()
                for msg in messages
            ]
            await asyncio.gather(*tasks)

            # Commit sau khi CẢ BATCH xử lý xong (lỗi từng job đã tự gửi error-result
            # về BE bên trong _dispatch_job, không throw ra đây) — nếu crash giữa batch,
            # message chưa commit sẽ được Kafka redeliver; các _process_* đều idempotent
            # (gửi lại cùng 1 kết quả cho BE) nên an toàn khi bị xử lý lại.
            await asyncio.wait_for(consumer.commit(), timeout=NETWORK_CALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            f"Kafka consumer hung for >{NETWORK_CALL_TIMEOUT_SECONDS}s (connection likely "
            "dead) — forcing reconnect"
        )
        raise
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped")


# Giới hạn số job chạy song song cùng lúc (tránh làm quá tải Rekognition/Milvus/S3).
# Đặt ở module-level để chặn đúng across nhiều batch liền kề, không chỉ trong 1 batch.
_JOB_SEMAPHORE = asyncio.Semaphore(8)


async def _dispatch_job(msg):
    """Route 1 Kafka message tới handler tương ứng, giới hạn concurrency bằng semaphore."""
    async with _JOB_SEMAPHORE:
        try:
            if msg.topic == TOPIC_PIPELINE_JOB:
                await _process_pipeline_job(msg.value)
            elif msg.topic == TOPIC_MODERATION_JOB:
                await _process_moderation_job(msg.value)
            elif msg.topic == TOPIC_RECOMMENDATION_SYNC:
                await _process_debezium_series(msg.value)
            elif msg.topic == TOPIC_MEDIA_DELETE:
                await _process_media_delete(msg.value)
        except Exception as e:
            logger.error(f"Job processing failed (topic={msg.topic}): {e}", exc_info=True)


async def _process_pipeline_job(data: dict):
    """Process a single pipeline job: download from S3, fingerprint, send result."""
    try:
        job = PipelineJobMessage.model_validate(data)
    except Exception as e:
        # Message sai schema (vd thiếu field bắt buộc) — phải gửi lỗi về BE ngay ở đây,
        # nếu không throw ra ngoài sẽ bị _dispatch_job() nuốt mất, BE không bao giờ nhận
        # được kết quả, media kẹt vĩnh viễn ở trạng thái đang xử lý.
        logger.error(f"Invalid pipeline job message: {e}")
        media_id = data.get("mediaId")
        if media_id:
            await send_copyright_result(media_id, {
                "mediaId": media_id,
                "correlationId": data.get("correlationId") or "",
                "contentId": None,
                "isDuplicate": False,
                "overallSimilarity": 0.0,
                "fingerprintCount": 0,
                "violations": [],
                "processedAt": datetime.utcnow().isoformat(),
                "success": False,
                "errorMessage": f"Invalid job message: {e}",
            })
        return
    logger.info(f"Processing pipeline job: mediaId={job.media_id}, type={job.media_type}")

    try:
        # Blocking S3/CPU work — run in a thread so it doesn't stall the event loop
        # and starve the Kafka consumer's heartbeat, which causes session timeouts
        # and endless redelivery of the same message.
        file_bytes = await asyncio.to_thread(download_from_s3, job.s3_key, job.s3_bucket)
        filename = job.s3_key.rsplit("/", 1)[-1] if "/" in job.s3_key else job.s3_key

        # Generate Preview
        preview_s3_key = None
        try:
            if job.media_type == "IMAGE":
                preview_bytes = generate_image_preview(file_bytes)
                preview_s3_key = f"images/previews/{job.media_id}.jpg"
                upload_to_s3(preview_s3_key, preview_bytes, content_type="image/jpeg", bucket=job.s3_bucket)
            elif job.media_type == "VIDEO":
                preview_bytes = generate_video_preview(file_bytes)
                preview_s3_key = f"videos/previews/{job.media_id}.mp4"
                upload_to_s3(preview_s3_key, preview_bytes, content_type="video/mp4", bucket=job.s3_bucket)
        except Exception as pe:
            logger.error(f"Failed to generate preview for {job.media_id}: {pe}")
            # We don't fail the entire job if preview fails

        # Run fingerprint pipeline
        response = await asyncio.to_thread(
            process_fingerprint, job.media_id, job.creator_id, file_bytes, filename
        )

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
            "previewS3Key": preview_s3_key,
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


async def _process_debezium_series(data: dict):
    """
    Process Debezium CDC event for public.series table.
    Expects data in standard Debezium format.
    """
    try:
        from app.kafka.kafka_producer_service import send_recommendation_result
        from app.services.recommendation_service import process_series_upsert, process_series_deletion, recalculate_series
        from app.db.mongodb import get_series_metadata

        payload = data.get("payload", data)
        op = payload.get("op", "u")
        
        if op == "c":
            logger.info("Ignored Create (op='c') event. Waiting for status change to SCHEDULED/PUBLISHED.")
            return
            
        if op == "d":
            before = payload.get("before")
            if not before:
                return
            series_id = before.get("series_id") or before.get("id")
            logger.info(f"Processing Deletion for series_id={series_id} (op='d')")
            
            neighbors = await asyncio.to_thread(process_series_deletion, str(series_id))
            await send_recommendation_result(str(series_id), [], action="DELETE")
            
            # Symmetric update for old neighbors
            for n_id in neighbors:
                new_sim = await asyncio.to_thread(recalculate_series, n_id)
                await send_recommendation_result(n_id, new_sim, action="UPSERT")
            return
            
        if op == "u" or op == "r":
            after = payload.get("after")
            if not after:
                return
                
            series_id = after.get("series_id") or after.get("id")
            title = after.get("title", "")
            description = after.get("description", "")
            status = after.get("status", "")
            is_deleted = after.get("is_deleted", False)
            
            # Treat specific statuses or is_deleted as Deletion
            if is_deleted or status in ["DELETED", "DRAFT", "HIDDEN", "FORCE_HIDDEN"]:
                logger.info(f"Processing Deletion for series_id={series_id} due to status={status}, is_deleted={is_deleted}")
                neighbors = await asyncio.to_thread(process_series_deletion, str(series_id))
                await send_recommendation_result(str(series_id), [], action="DELETE")
                
                # Symmetric update for old neighbors
                for n_id in neighbors:
                    new_sim = await asyncio.to_thread(recalculate_series, n_id)
                    await send_recommendation_result(n_id, new_sim, action="UPSERT")
                return
                
            # Process Upsert only for valid active statuses
            if status in ["PUBLISHED", "SCHEDULED"]:
                logger.info(f"Processing Upsert for series_id={series_id} due to status={status}")
                metadata = await get_series_metadata(str(series_id))
                categories = metadata.get("category", [])
                tags = metadata.get("tags", [])
                age_rating = metadata.get("ageRating", "")
                language = metadata.get("language", "")
                
                similar_ids = await asyncio.to_thread(process_series_upsert, str(series_id), title, description, categories, tags, age_rating, language)
                await send_recommendation_result(str(series_id), similar_ids, action="UPSERT")
                
                # Symmetric update for new neighbors
                for n_id in similar_ids:
                    new_sim = await asyncio.to_thread(recalculate_series, n_id)
                    await send_recommendation_result(n_id, new_sim, action="UPSERT")
            else:
                logger.info(f"Ignored Upsert for series_id={series_id}, unsupported status={status}")

    except Exception as e:
        logger.error(f"Failed to process Debezium series event: {e}", exc_info=True)


async def _process_moderation_job(data: dict):
    """Process moderation job: download from S3, run Rekognition, send result."""
    try:
        job = PipelineJobMessage.model_validate(data)
    except Exception as e:
        # Xem giải thích ở _process_pipeline_job() — không được để lỗi validate văng ra
        # ngoài, nếu không job mất tích vĩnh viễn, BE không bao giờ nhận được kết quả.
        logger.error(f"Invalid moderation job message: {e}")
        media_id = data.get("mediaId")
        if media_id:
            await send_moderation_result(media_id, {
                "mediaId": media_id,
                "correlationId": data.get("correlationId") or "",
                "isSafe": False,
                "primaryLabel": None,
                "confidenceScore": 0.0,
                "violations": [],
                "rawResponse": "",
                "processedAt": datetime.utcnow().isoformat(),
                "success": False,
                "errorMessage": f"Invalid job message: {e}",
            })
        return
    logger.info(f"Processing moderation job: mediaId={job.media_id}, type={job.media_type}")

    try:
        # Blocking S3/Rekognition work — run in a thread, same reasoning as the
        # copyright pipeline (avoid starving the Kafka consumer's heartbeat).
        file_bytes = await asyncio.to_thread(download_from_s3, job.s3_key, job.s3_bucket)
        result = await asyncio.to_thread(
            moderate_media, file_bytes, job.media_type, job.media_id, job.correlation_id
        )
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


async def _process_media_delete(data: dict):
    """Dọn fingerprint Milvus khi BE báo media đã bị xóa (best-effort, không có result topic)."""
    media_id = data.get("mediaId")
    if not media_id:
        return
    try:
        await asyncio.to_thread(delete_fingerprint, media_id)
        logger.info(f"Deleted fingerprint for mediaId={media_id}")
    except Exception as e:
        logger.error(f"Failed to delete fingerprint for mediaId={media_id}: {e}")
