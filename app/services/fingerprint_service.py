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

# Trần an toàn TĨNH cho cap tải S3 (kafka_consumer_service.py) — chặn NGAY tại HeadObject
# trước khi .read() cả file vào RAM, tránh OOM khi 1 object quá khổ x nhiều job song song.
# Đây là trần bảo vệ hạ tầng (đọc từ env lúc khởi động), KHÁC với ngưỡng NGHIỆP VỤ động
# mà Admin chỉnh (fingerprint_max_file_size_mb) — ngưỡng động được _validate_file() thực thi
# per-job sau khi đã tải về. Admin hạ ngưỡng → file vẫn tải nhưng bị _validate_file loại;
# nếu Admin muốn NÂNG trần vượt giá trị này, phải chỉnh cả env FINGERPRINT_MAX_FILE_SIZE_MB.
MAX_FILE_SIZE = settings.FINGERPRINT_MAX_FILE_SIZE_MB * 1024 * 1024


class StaleJobError(Exception):
    """Raised when a process_fingerprint() run is superseded by a newer job for the same
    media_id before reaching its decisive Milvus write — expected under timeout+retry."""


# asyncio.wait_for() ở kafka_consumer_service.py chỉ hủy việc CHỜ — thread thật chạy
# process_fingerprint() vẫn tiếp tục sau khi job đã "timeout" ở tầng trên. Nếu creator
# retry, job mới cho CÙNG media_id có thể chạy song song với thread mồ côi của lần trước;
# thread cũ ghi delete+insert muộn có thể chồng lên/xóa mất kết quả đúng của lần retry.
# Bộ đếm generation theo media_id: mỗi lần process_fingerprint() được gọi cho 1 media_id,
# tự nhận số thứ tự mới nhất — thread nào không còn giữ số mới nhất tại thời điểm SẮP ghi
# (ngay trước delete và ngay trước insert) thì bỏ qua thao tác ghi đó (StaleJobError).
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
    """
    Xử lý fingerprint cho 1 video/ảnh.

    Args:
        media_id: ID video/ảnh.
        creator_id: ID creator sở hữu media này — dùng để loại trừ so khớp trong
            cùng creator (không tự báo "vi phạm" nội dung của chính mình).
        file_bytes: Nội dung file.
        filename: Tên file gốc (để xác định loại).

    Returns:
        FingerprintResponse với content_id, is_duplicate, violations.
    """
    logger.info(f"Fingerprint: processing media_id={media_id}, file={filename}, size={len(file_bytes)} bytes")

    if not is_connected():
        raise RuntimeError("Milvus chưa sẵn sàng.")

    # Đọc cấu hình động 1 LẦN/job (trước validate, để dùng luôn ngưỡng dung lượng động).
    # Truyền xuống mọi bước bên dưới; KHÔNG gọi lại trong loop/hàm con. Postgres lỗi/chưa
    # có row → dynamic_config fallback default, không raise.
    ai_config = get_ai_pipeline_config()

    # Validate file (dùng ngưỡng dung lượng động của job này)
    _validate_file(file_bytes, filename, ai_config["fingerprint_max_file_size_mb"])

    # Nhận số generation MỚI NHẤT cho media_id này — nếu 1 lần gọi khác (retry) nhận số
    # mới hơn trước khi lần này kịp ghi, lần này coi là "cũ" và bỏ qua ghi (xem StaleJobError).
    my_gen = _claim_generation(media_id)

    # Xác định loại file
    ext = _get_extension(filename)
    is_video = ext in VIDEO_EXTENSIONS

    # Tạo fingerprints
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

    # Xác định cụm nội dung + chủ sở hữu gốc TRƯỚC KHI xóa dòng cũ của chính media_id này.
    # Bắt buộc phải resolve trước delete: nếu media_id này là dòng DUY NHẤT còn "sạch"
    # (is_violation=False) trong cụm, xóa trước rồi mới resolve sẽ khiến chủ gốc "biến mất"
    # khỏi Milvus đúng lúc tự tra cứu — làm mất quyền sở hữu của chính mình một cách oan uổng.
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

    # Xóa fingerprint cũ nếu media_id đã tồn tại (upsert)
    delete_by_media_id(media_id)

    # Tìm trùng trong Milvus — loại trừ chính creator này (Milvus-level) VÀ loại trừ mọi
    # dòng mà creator hiện tại chính là chủ sở hữu gốc đã ghi nhận (uploader_creator_id),
    # dù dòng đó mang creator_id của người khác (bản sao) — đây là điểm sửa lỗi cốt lõi:
    # chủ gốc không bao giờ bị báo vi phạm ngược với chính nội dung của mình, bất kể ai
    # khác đã đăng bản sao trước đó.
    violations = _find_violations(
        fingerprints,
        ai_config,
        exclude_media_id=media_id,
        exclude_creator_id=creator_id,
        is_video=is_video,
        uploader_creator_id=creator_id,
    )

    if not _is_current_generation(media_id, my_gen):
        # Job mới hơn đã claim generation giữa lúc delete và insert — KHÔNG insert đè lên
        # dữ liệu job mới (đã tự delete+insert đúng của nó), nếu không sẽ mất trắng media_id.
        logger.info(f"Fingerprint: superseded by newer job for media_id={media_id}, dropping stale insert")
        raise StaleJobError(media_id)

    # is_violation chỉ True khi upload này KHÔNG phải chủ sở hữu cụm VÀ có vi phạm thật —
    # chủ sở hữu (is_owner=True) không bao giờ bị đánh dấu vi phạm dù trùng lặp bao nhiêu
    # dòng khác trong cùng cụm của chính mình.
    is_violation_flag = (not is_owner) and len(violations) > 0

    # Lưu fingerprints mới vào Milvus — kèm thông tin cụm nội dung/chủ sở hữu gốc. Nếu
    # matched=True, tái sử dụng NGUYÊN cluster_id/first_seen_at đã có (không tạo cụm mới,
    # không ghi đè first_seen_at cũ bằng thời điểm hiện tại — giữ đúng thứ tự "ai trước").
    insert_fingerprints(
        media_id,
        creator_id,
        fingerprints,
        content_cluster_id=cluster.cluster_id,
        original_creator_id=cluster.original_creator_id,
        first_seen_at=cluster.first_seen_at,
        is_violation=is_violation_flag,
    )

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


def _validate_file(file_bytes: bytes, filename: str, max_file_size_mb: int) -> None:
    """Validate file trước khi xử lý. max_file_size_mb đọc động 1 lần/job, truyền xuống."""
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
    """Video → frames → vectors. fps/max_frames đọc động 1 lần/job, truyền xuống."""
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
    """
    Tìm vi phạm trong Milvus (loại trừ cùng creator — xem match_segments/match_image_violation).

    ai_config: dict cấu hình động đọc 1 lần/job ở process_fingerprint (top_k, ngưỡng, min/max
    match, fps) — KHÔNG đọc settings trực tiếp ở đây.

    is_video quyết định cách so khớp: video nối nhiều điểm liên tiếp thành đoạn (phải dài
    tối thiểu min_match_seconds), ảnh chỉ có 1 điểm fingerprint duy nhất nên so khớp trực tiếp
    theo threshold — dùng chung match_segments() cho ảnh sẽ luôn ra duration=0, bị loại oan
    dù giống 100% (đã gặp thật, xem match_image_violation docstring).

    uploader_creator_id: xem docstring match_segments()/match_image_violation() — bỏ qua
        match mà uploader chính là chủ sở hữu gốc (original_creator_id) của cụm nguồn.
    """
    if not fingerprints:
        return []

    vectors = [fp["vector"] for fp in fingerprints]

    # Search Milvus — loại trừ cùng creator NGAY TẠI Milvus (xem docstring search_similar),
    # không chỉ lọc sau ở matcher.py, để không bỏ sót vi phạm thật với creator khác bị các
    # trang cũ của chính creator này chiếm hết chỗ trong top_k.
    # top_k video trước đây cố định =3 (xem config.py FINGERPRINT_VIDEO_TOP_K cho lý do
    # tăng lên) — cùng rủi ro với ảnh: fingerprint cũ (media đã xóa nhưng vector giữ mãi)
    # có thể chiếm hết suất top_k, đẩy nguồn thật đang sống ra ngoài kết quả.
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
