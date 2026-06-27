"""AWS Rekognition client — DetectModerationLabels for content safety."""

import boto3
from loguru import logger
from app.core.config import settings

_rek_client = None


def get_rekognition_client():
    """Lazy-init Rekognition client."""
    global _rek_client
    if _rek_client is None and settings.AWS_ACCESS_KEY_ID:
        _rek_client = boto3.client(
            "rekognition",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
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
