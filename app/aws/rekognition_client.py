"""AWS Rekognition client — DetectModerationLabels for content safety."""

import boto3
from botocore.config import Config
from loguru import logger
from app.core.config import settings

_rek_client = None

# Xem giải thích chi tiết ở s3_client.py — timeout job-level (asyncio.wait_for) không ép
# dừng được thread thật, nên rút ngắn timeout boto3 để request tự bỏ cuộc nhanh hơn,
# tránh thread "mắc kẹt" chiếm thread pool dùng chung nhiều phút sau khi job đã timeout.
_BOTO_CONFIG = Config(connect_timeout=10, read_timeout=20, retries={"max_attempts": 2, "mode": "standard"})


def get_rekognition_client():
    """Lazy-init Rekognition client."""
    global _rek_client
    if _rek_client is None and settings.AWS_ACCESS_KEY_ID:
        _rek_client = boto3.client(
            "rekognition",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            config=_BOTO_CONFIG,
        )
    return _rek_client


def detect_moderation_labels(image_bytes: bytes) -> list[dict]:
    """Call Rekognition DetectModerationLabels on a single image.
    Returns list of {name, confidence, parent_name}."""
    client = get_rekognition_client()
    if client is None:
        raise RuntimeError("Rekognition client not configured — check AWS credentials")

    response = client.detect_moderation_labels(
        Image={"Bytes": image_bytes},
        MinConfidence=50.0,
    )
    labels = []
    for label in response.get("ModerationLabels", []):
        labels.append({
            "name": label["Name"],
            "confidence": label["Confidence"],
            "parent_name": label.get("ParentName", ""),
        })
    return labels
