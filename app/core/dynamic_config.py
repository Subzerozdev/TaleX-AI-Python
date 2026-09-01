
import logging
import os

import psycopg2

from app.core.config import settings

logger = logging.getLogger(__name__)

# Cột trong bảng ai_pipeline_configs — (tên_cột, kiểu ép). Tên PHẢI khớp entity Java
# (AiPipelineConfig). Kiểu ép quan trọng: top_k/fps/frames là int (Milvus/phép chia frame
# cần int), các ngưỡng/interval là float.
_COLUMNS = (
    ("fingerprint_similarity_threshold", float),
    ("fingerprint_cluster_threshold", float),
    ("rekognition_confidence_threshold", float),
    ("rekognition_violence_confidence_threshold", float),
    ("fingerprint_image_top_k", int),
    ("fingerprint_video_top_k", int),
    ("fingerprint_min_match_seconds", int),
    ("fingerprint_max_gap_seconds", int),
    ("fingerprint_fps", int),
    ("fingerprint_max_frames", int),
    ("fingerprint_max_file_size_mb", int),
    ("rekognition_max_frames", int),
    ("moderation_frame_interval", float),
)


def _defaults() -> dict:
    """Fallback lấy từ config.py (env-var static) — không hardcode lại số ở đây."""
    return {
        "fingerprint_similarity_threshold": settings.FINGERPRINT_SIMILARITY_THRESHOLD,
        "fingerprint_cluster_threshold": settings.FINGERPRINT_CLUSTER_THRESHOLD,
        "rekognition_confidence_threshold": settings.REKOGNITION_CONFIDENCE_THRESHOLD,
        "rekognition_violence_confidence_threshold": settings.REKOGNITION_VIOLENCE_CONFIDENCE_THRESHOLD,
        "fingerprint_image_top_k": settings.FINGERPRINT_IMAGE_TOP_K,
        "fingerprint_video_top_k": settings.FINGERPRINT_VIDEO_TOP_K,
        "fingerprint_min_match_seconds": settings.FINGERPRINT_MIN_MATCH_SECONDS,
        "fingerprint_max_gap_seconds": settings.FINGERPRINT_MAX_GAP_SECONDS,
        "fingerprint_fps": settings.FINGERPRINT_FPS,
        "fingerprint_max_frames": settings.FINGERPRINT_MAX_FRAMES,
        "fingerprint_max_file_size_mb": settings.FINGERPRINT_MAX_FILE_SIZE_MB,
        "rekognition_max_frames": settings.REKOGNITION_MAX_FRAMES,
        "moderation_frame_interval": settings.MODERATION_FRAME_INTERVAL,
    }


def get_ai_pipeline_config() -> dict:
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
        )
        column_names = ", ".join(name for name, _ in _COLUMNS)
        with conn.cursor() as cur:
            cur.execute(f"SELECT {column_names} FROM ai_pipeline_configs LIMIT 1")
            row = cur.fetchone()

        if row is None:
            logger.warning(
                "ai_pipeline_configs chưa có row nào — dùng default từ config.py"
            )
            return _defaults()

        return {name: caster(val) for (name, caster), val in zip(_COLUMNS, row)}

    except Exception as e:
        logger.warning(
            "Không đọc được ai_pipeline_configs (%s) — dùng default từ config.py", e
        )
        return _defaults()
    finally:
        if conn is not None:
            conn.close()
