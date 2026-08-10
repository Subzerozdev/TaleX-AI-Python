"""S3 download client for content pipeline."""

import boto3
from botocore.config import Config
from loguru import logger
from app.core.config import settings

_s3_client = None

# asyncio.wait_for(timeout=60s) bọc quanh mỗi job (kafka_consumer_service.py) chỉ hủy
# việc CHỜ — không ép dừng được thread thật đang chạy lời gọi boto3 bên dưới. Mặc định
# boto3 là connect_timeout=60s + read_timeout=60s MỖI request, cộng thêm vài lần retry
# tự động, nên 1 request thật sự bị treo có thể chạy ngầm nhiều phút sau khi job đã
# "timeout" ở tầng trên — nhiều job dồn lại kiểu này có thể chiếm hết thread pool mặc
# định (dùng chung cho MỌI asyncio.to_thread trong app). Set timeout ngắn hơn ở đây để
# request tự bỏ cuộc nhanh, giảm thời gian thread bị "mắc kẹt".
_BOTO_CONFIG = Config(connect_timeout=10, read_timeout=20, retries={"max_attempts": 2, "mode": "standard"})


def get_s3_client():
    """Lazy-init S3 client. Returns None if credentials missing."""
    global _s3_client
    if _s3_client is None and settings.AWS_ACCESS_KEY_ID:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            config=_BOTO_CONFIG,
        )
    return _s3_client


def download_from_s3(s3_key: str, bucket: str = None, max_bytes: int | None = None) -> bytes:
    """Download file from S3, return bytes."""
    bucket = bucket or settings.AWS_S3_BUCKET
    client = get_s3_client()
    if client is None:
        raise RuntimeError("S3 client not configured — check AWS credentials")
    if max_bytes is not None:
        # Chặn TRƯỚC khi tải — .read() không giới hạn dung lượng có thể đọc cả file vài
        # GB thẳng vào RAM; với nhiều job chạy song song (_JOB_SEMAPHORE), vài file lớn
        # cùng lúc đủ để OOM cả service. HeadObject rẻ (chỉ lấy metadata, không tải body).
        head = client.head_object(Bucket=bucket, Key=s3_key)
        content_length = head.get("ContentLength")
        if content_length is not None and content_length > max_bytes:
            raise ValueError(
                f"S3 object {s3_key} is {content_length} bytes, exceeds cap {max_bytes} bytes"
            )
        if content_length is None:
            logger.warning(f"S3 HeadObject for {s3_key} missing ContentLength — proceeding without pre-check")
    logger.info(f"S3 download: bucket={bucket}, key={s3_key}")
    response = client.get_object(Bucket=bucket, Key=s3_key)
    data = response["Body"].read()
    logger.info(f"S3 download complete: {len(data)} bytes")
    return data

def upload_to_s3(s3_key: str, file_bytes: bytes, content_type: str = "application/octet-stream", bucket: str = None) -> None:
    """Upload file bytes to S3."""
    bucket = bucket or settings.AWS_S3_BUCKET
    client = get_s3_client()
    if client is None:
        raise RuntimeError("S3 client not configured — check AWS credentials")
    logger.info(f"S3 upload: bucket={bucket}, key={s3_key}, size={len(file_bytes)} bytes")
    # Thêm CacheControl='no-transform' để ngăn Cloudflare Polish hoặc CDN nén lại ảnh (làm hỏng watermark)
    client.put_object(
        Bucket=bucket, 
        Key=s3_key, 
        Body=file_bytes, 
        ContentType=content_type,
        CacheControl="no-transform"
    )
    logger.info(f"S3 upload complete: {s3_key}")
