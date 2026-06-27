"""S3 download client for content pipeline."""

import boto3
from loguru import logger
from app.core.config import settings

_s3_client = None


def get_s3_client():
    """Lazy-init S3 client. Returns None if credentials missing."""
    global _s3_client
    if _s3_client is None and settings.AWS_ACCESS_KEY_ID:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
    return _s3_client


def download_from_s3(s3_key: str, bucket: str = None) -> bytes:
    """Download file from S3, return bytes."""
    bucket = bucket or settings.AWS_S3_BUCKET
    client = get_s3_client()
    if client is None:
        raise RuntimeError("S3 client not configured — check AWS credentials")
    logger.info(f"S3 download: bucket={bucket}, key={s3_key}")
    response = client.get_object(Bucket=bucket, Key=s3_key)
    data = response["Body"].read()
    logger.info(f"S3 download complete: {len(data)} bytes")
    return data
