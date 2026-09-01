

import threading

from loguru import logger

from app.core.config import settings
from app.core.dynamic_config import get_ai_pipeline_config
from app.fingerprint.content_ownership import resolve_content_cluster
from app.fingerprint.extractor import extract_frames_from_video, extract_image
from app.fingerprint.hasher import hash_frames, hash_image, VECTOR_DIM
from app.fingerprint.matcher import match_image_violation, match_segments
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

MAX_FILE_SIZE = settings.FINGERPRINT_MAX_FILE_SIZE_MB * 1024 * 1024


class StaleJobError(Exception):
    """Raised when a process_fingerprint() run is superseded by a newer job for the same
    media_id before reaching its decisive Milvus write — expected under timeout+retry."""

_media_generation: dict[str, int] = {}
_media_gen_lock = threading.Lock()


def _claim_generation(media_id: str) -> int:
    with _media_gen_lock:
        gen = _media_generation.get(media_id, 0) + 1
        _media_generation[media_id] = gen
        return gen


def _is_current_generation(media_id: str, my_gen: int) -> bool:
    with _media_gen_lock:
        return _media_generation.get(media_id) == my_gen


def process_fingerprint(
    media_id: str, creator_id: str, file_bytes: bytes, filename: str
) -> FingerprintResponse:
    logger.info(f"Fingerprint: processing media_id={media_id}, file={filename}, size={len(file_bytes)} bytes")

    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")


    ai_config = get_ai_pipeline_config()
    _validate_file(file_bytes, filename, ai_config["fingerprint_max_file_size_mb"])
    my_gen = _claim_generation(media_id)

    ext = _get_extension(filename)
    is_video = ext in VIDEO_EXTENSIONS

    if is_video:
        fingerprints = _process_video(
            file_bytes,
            fps=ai_config["fingerprint_fps"],
            max_frames=ai_config["fingerprint_max_frames"],
        )
    else:
        fingerprints = _process_image(file_bytes)

    if not _is_current_generation(media_id, my_gen):
        logger.info(f"Fingerprint: superseded by newer job for media_id={media_id}, dropping stale write")
        raise StaleJobError(media_id)

    vectors = [fp["vector"] for fp in fingerprints]
    cluster = resolve_content_cluster(
        vectors,
        creator_id=creator_id,
        is_video=is_video,
        cluster_threshold=ai_config["fingerprint_cluster_threshold"],
        image_top_k=ai_config["fingerprint_image_top_k"],
        video_top_k=ai_config["fingerprint_video_top_k"],
        min_match_seconds=ai_config["fingerprint_min_match_seconds"],
        fps=ai_config["fingerprint_fps"],
    )
    is_owner = cluster.matched and cluster.original_creator_id == creator_id

    delete_by_media_id(media_id)

    violations = _find_violations(
        fingerprints,
        ai_config,
        exclude_media_id=media_id,
        exclude_creator_id=creator_id,
        is_video=is_video,
        uploader_creator_id=creator_id,
    )

    if not _is_current_generation(media_id, my_gen):
        logger.info(f"Fingerprint: superseded by newer job for media_id={media_id}, dropping stale insert")
        raise StaleJobError(media_id)

    is_violation_flag = (not is_owner) and len(violations) > 0

    insert_fingerprints(
        media_id,
        creator_id,
        fingerprints,
        content_cluster_id=cluster.cluster_id,
        original_creator_id=cluster.original_creator_id,
        first_seen_at=cluster.first_seen_at,
        is_violation=is_violation_flag,
    )

    content_id = f"CID-{media_id}"

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

    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    content_id = f"CID-{media_id}"

    return FingerprintInfo(
        media_id=media_id,
        content_id=content_id,
        fingerprint_count=0,  # sẽ cải thiện sau
        is_stored=True,
    )


def delete_fingerprint(media_id: str) -> DeleteResponse:
    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    count = delete_by_media_id(media_id)

    return DeleteResponse(
        success=True,
        media_id=media_id,
        deleted_count=count,
        message=f"Đã xóa {count} fingerprints của media_id={media_id}.",
    )


def _validate_file(file_bytes: bytes, filename: str, max_file_size_mb: int) -> None:

    if len(file_bytes) == 0:
        raise ValueError("File rỗng.")

    max_bytes = max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(f"File quá lớn ({size_mb:.1f}MB). Giới hạn: {max_file_size_mb}MB.")

    ext = _get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Không hỗ trợ file {ext}. Chỉ hỗ trợ: {', '.join(sorted(ALLOWED_EXTENSIONS))}")


def _get_extension(filename: str) -> str:
    """Lấy extension từ filename (lowercase)."""
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _process_video(file_bytes: bytes, fps: int, max_frames: int) -> list[dict]:
    frames = extract_frames_from_video(file_bytes, fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError("Không thể trích xuất frames từ video. File có thể bị hỏng.")
    return hash_frames(frames)


def _process_image(file_bytes: bytes) -> list[dict]:
    """Image → vector."""
    image = extract_image(file_bytes)
    vector = hash_image(image)
    return [{"timestamp": 0.0, "vector": vector}]


def _find_violations(
    fingerprints: list[dict],
    ai_config: dict,
    exclude_media_id: str,
    exclude_creator_id: str = "",
    is_video: bool = True,
    uploader_creator_id: str | None = None,
) -> list[dict]:
    if not fingerprints:
        return []

    vectors = [fp["vector"] for fp in fingerprints]

    top_k = (
        ai_config["fingerprint_video_top_k"]
        if is_video
        else ai_config["fingerprint_image_top_k"]
    )
    search_results = search_similar(vectors, top_k=top_k, exclude_creator_id=exclude_creator_id or None)

    if not search_results:
        return []

    if is_video:
        return match_segments(
            query_fingerprints=fingerprints,
            search_results=search_results,
            similarity_threshold=ai_config["fingerprint_similarity_threshold"],
            min_match_seconds=ai_config["fingerprint_min_match_seconds"],
            max_gap_seconds=ai_config["fingerprint_max_gap_seconds"],
            fps=ai_config["fingerprint_fps"],
            exclude_media_id=exclude_media_id,
            exclude_creator_id=exclude_creator_id,
            uploader_creator_id=uploader_creator_id,
        )

    return match_image_violation(
        search_results=search_results,
        similarity_threshold=ai_config["fingerprint_similarity_threshold"],
        exclude_media_id=exclude_media_id,
        exclude_creator_id=exclude_creator_id,
        uploader_creator_id=uploader_creator_id,
    )
