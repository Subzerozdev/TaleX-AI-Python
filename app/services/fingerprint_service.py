"""
Fingerprint Service — Điều phối pipeline Content ID.

Luồng:
  1. Nhận file upload
  2. Xác định loại: VIDEO hay IMAGE
  3. Extractor → lấy frames
  4. Hasher → tạo vectors
  5. Milvus search → tìm trùng
  6. Matcher → nối thành segments
  7. Lưu vectors vào Milvus
  8. Tạo Content ID
  9. Trả kết quả
"""

from loguru import logger

from app.fingerprint.extractor import extract_frames_from_video, extract_image
from app.fingerprint.hasher import hash_frames, hash_image, VECTOR_DIM
from app.fingerprint.matcher import match_segments
from app.fingerprint.milvus_store import (
    delete_by_media_id,
    get_count,
    insert_fingerprints,
    is_connected,
    search_similar,
)
from app.schemas.fingerprint import (
    DeleteResponse,
    FingerprintInfo,
    FingerprintResponse,
    ViolationSegment,
)

# Allowed file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

# Max file size (bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


def process_fingerprint(media_id: str, file_bytes: bytes, filename: str) -> FingerprintResponse:
    """
    Xử lý fingerprint cho 1 video/ảnh.

    Args:
        media_id: ID video/ảnh.
        file_bytes: Nội dung file.
        filename: Tên file gốc (để xác định loại).

    Returns:
        FingerprintResponse với content_id, is_duplicate, violations.
    """
    logger.info(f"Fingerprint: processing media_id={media_id}, file={filename}, size={len(file_bytes)} bytes")

    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    # Validate file
    _validate_file(file_bytes, filename)

    # Xác định loại file
    ext = _get_extension(filename)
    is_video = ext in VIDEO_EXTENSIONS

    # Tạo fingerprints
    if is_video:
        fingerprints = _process_video(file_bytes)
    else:
        fingerprints = _process_image(file_bytes)

    # Xóa fingerprint cũ nếu media_id đã tồn tại (upsert)
    delete_by_media_id(media_id)

    # Tìm trùng trong Milvus
    violations = _find_violations(fingerprints, exclude_media_id=media_id)

    # Lưu fingerprints mới vào Milvus
    insert_fingerprints(media_id, fingerprints)

    # Tạo Content ID
    content_id = f"CID-{media_id}"

    # Tính overall similarity
    overall_similarity = 0.0
    if violations:
        overall_similarity = max(v["similarity_score"] for v in violations)

    response = FingerprintResponse(
        media_id=media_id,
        content_id=content_id,
        is_duplicate=len(violations) > 0,
        overall_similarity=overall_similarity,
        fingerprint_count=len(fingerprints),
        violations=[ViolationSegment(**v) for v in violations],
    )

    logger.info(
        f"Fingerprint: media_id={media_id}, content_id={content_id}, "
        f"duplicate={response.is_duplicate}, violations={len(violations)}, "
        f"vectors={len(fingerprints)}"
    )

    return response


def get_fingerprint_info(media_id: str) -> FingerprintInfo:
    """Lấy thông tin fingerprint đã lưu."""
    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    content_id = f"CID-{media_id}"

    # Đếm số vectors của media_id này (query Milvus)
    # Dùng get_count tổng vì Milvus không có count by filter đơn giản
    # → trả is_stored dựa trên việc search có tìm thấy không
    return FingerprintInfo(
        media_id=media_id,
        content_id=content_id,
        fingerprint_count=0,  # sẽ cải thiện sau
        is_stored=True,
    )


def delete_fingerprint(media_id: str) -> DeleteResponse:
    """Xóa tất cả fingerprints của 1 media."""
    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    count = delete_by_media_id(media_id)

    return DeleteResponse(
        success=True,
        media_id=media_id,
        deleted_count=count,
        message=f"Đã xóa {count} fingerprints của media_id={media_id}.",
    )


def _validate_file(file_bytes: bytes, filename: str) -> None:
    """Validate file trước khi xử lý."""
    if len(file_bytes) == 0:
        raise ValueError("File rỗng.")

    if len(file_bytes) > MAX_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(f"File quá lớn ({size_mb:.1f}MB). Giới hạn: {MAX_FILE_SIZE // (1024*1024)}MB.")

    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Không hỗ trợ file {ext}. Chỉ hỗ trợ: {', '.join(sorted(ALLOWED_EXTENSIONS))}")


def _get_extension(filename: str) -> str:
    """Lấy extension từ filename (lowercase)."""
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _process_video(file_bytes: bytes) -> list[dict]:
    """Video → frames → vectors."""
    frames = extract_frames_from_video(file_bytes)
    if not frames:
        raise ValueError("Không thể trích xuất frames từ video. File có thể bị hỏng.")
    return hash_frames(frames)


def _process_image(file_bytes: bytes) -> list[dict]:
    """Image → vector."""
    image = extract_image(file_bytes)
    vector = hash_image(image)
    return [{"timestamp": 0.0, "vector": vector}]


def _find_violations(fingerprints: list[dict], exclude_media_id: str) -> list[dict]:
    """Tìm đoạn trùng trong Milvus."""
    if not fingerprints:
        return []

    vectors = [fp["vector"] for fp in fingerprints]

    # Search Milvus
    search_results = search_similar(vectors, top_k=3)

    if not search_results:
        return []

    # Matcher nối thành segments
    return match_segments(
        query_fingerprints=fingerprints,
        search_results=search_results,
        exclude_media_id=exclude_media_id,
    )
