"""
Ownership — Sổ đăng ký chủ sở hữu nội dung gốc (content-ownership registry).

Xác định 2 việc:
  1. resolve_content_cluster(): upload mới này thuộc cụm nội dung nào đã có trong hệ thống
     (nếu có), và ai là chủ sở hữu gốc (original_creator_id) của cụm đó — theo nguyên tắc
     "ai upload trước, người đó là chủ" (first-seen-wins).
  2. get_cluster_owner(): tra lại chủ sở hữu của 1 cụm ĐÃ BIẾT content_cluster_id — dùng khi
     process_fingerprint() cần xác nhận lại (fallback) từ các dòng còn lại trong Milvus.

QUAN TRỌNG: search ở đây KHÔNG loại trừ creator hiện tại (khác _find_violations() ở
fingerprint_service.py) — vì mục đích là "tìm đúng cụm nội dung", và nếu chủ gốc re-upload
lại nội dung của chính mình thì upload đó PHẢI khớp lại với cụm cũ của chính họ. Đây chính là
điểm khác biệt cốt lõi giải quyết bug: chủ gốc không bao giờ bị loại khỏi việc tự nhận diện
cụm nội dung của chính mình, dù bao nhiêu bản sao của creator khác đang có trong Milvus.
"""

import time
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.fingerprint.milvus_store import query_cluster_rows, search_similar


@dataclass
class ClusterResolution:
    """Kết quả tra cứu cụm nội dung cho 1 upload/lần tra cứu."""

    matched: bool
    cluster_id: str
    original_creator_id: str
    first_seen_at: float


def resolve_content_cluster(
    vectors: list[bytes], creator_id: str, is_video: bool
) -> ClusterResolution:
    """
    Xác định cụm nội dung + chủ sở hữu gốc cho 1 upload mới.

    Nếu khớp với cụm đã có → trả về cluster_id/original_creator_id/first_seen_at của cụm đó
    (chủ gốc luôn được xác nhận lại đúng, bất kể bao nhiêu bản sao của creator khác đang có
    trong Milvus). Nếu không khớp cụm nào → tạo cụm mới, chủ sở hữu = creator đang upload.

    Args:
        vectors: Fingerprint vectors của upload (1 vector cho ảnh, nhiều cho video).
        creator_id: Creator đang thực hiện upload — chủ sở hữu MẶC ĐỊNH nếu là cụm mới.
        is_video: True → dùng thuật toán tổng hợp nhiều frame; False → ảnh (1 vector).

    Returns:
        ClusterResolution — matched=True nếu khớp cụm cũ, False nếu tự mint cụm mới. Caller
        (process_fingerprint) dùng cluster_id/original_creator_id/first_seen_at trong cả 2
        trường hợp để insert; "matched" chỉ là cờ thông tin.
    """
    if not vectors:
        return _mint_new_cluster(creator_id)

    top_k = settings.FINGERPRINT_IMAGE_TOP_K if not is_video else 3
    # KHÔNG loại trừ creator hiện tại — xem docstring module.
    raw_matches = search_similar(vectors, top_k=top_k, exclude_creator_id=None)

    candidates = [
        m
        for m in raw_matches
        if m["score"] >= settings.FINGERPRINT_CLUSTER_THRESHOLD
        and not m["is_violation"]
        and m["content_cluster_id"]
    ]

    if not candidates:
        return _mint_new_cluster(creator_id)

    if is_video:
        chosen = _pick_cluster_by_coverage(candidates)
        if chosen is None:
            return _mint_new_cluster(creator_id)
    else:
        chosen = min(candidates, key=lambda m: (m["first_seen_at"], m["media_id"]))

    return ClusterResolution(
        matched=True,
        cluster_id=chosen["content_cluster_id"],
        original_creator_id=chosen["original_creator_id"],
        first_seen_at=chosen["first_seen_at"],
    )


def get_cluster_owner(content_cluster_id: str) -> ClusterResolution | None:
    """
    Tra lại chủ sở hữu của 1 cụm nội dung ĐÃ BIẾT content_cluster_id.

    Dùng làm fallback trong process_fingerprint() để xác nhận lại chủ sở hữu từ các dòng
    còn lại trong Milvus — ví dụ ngay sau khi xóa media_id cũ của chính chủ sở hữu (upsert),
    khi resolve_content_cluster() ở lần gọi kế tiếp cần "nhớ lại" đúng cluster_id đã biết
    thay vì search lại từ đầu.

    Args:
        content_cluster_id: ID cụm đã biết từ 1 lần resolve trước đó.

    Returns:
        ClusterResolution (matched=True) nếu cụm còn dòng "sạch" (is_violation=False), None
        nếu cụm không còn dòng nào (toàn bộ dòng sạch của cụm đã bị xóa).
    """
    rows = query_cluster_rows(content_cluster_id, only_non_violation=True)
    if not rows:
        return None
    earliest = min(rows, key=lambda r: r["first_seen_at"])
    return ClusterResolution(
        matched=True,
        cluster_id=content_cluster_id,
        original_creator_id=earliest["original_creator_id"],
        first_seen_at=earliest["first_seen_at"],
    )


def _mint_new_cluster(creator_id: str) -> ClusterResolution:
    """Tạo cụm nội dung MỚI — creator đang upload trở thành chủ sở hữu gốc."""
    return ClusterResolution(
        matched=False,
        cluster_id=str(uuid.uuid4()),
        original_creator_id=creator_id,
        first_seen_at=time.time(),
    )


def _pick_cluster_by_coverage(candidates: list[dict]) -> dict | None:
    """
    Video: gom candidate theo content_cluster_id, chọn cụm có nhiều query frame khớp nhất
    (coverage). Yêu cầu tối thiểu FINGERPRINT_MIN_MATCH_SECONDS (quy đổi theo FPS thành số
    frame) khớp mới coi là "cùng nội dung" — tránh 1-2 frame trùng ngẫu nhiên (ví dụ màu nền
    giống nhau) bị coi là cùng cụm với cả video. Đây chỉ là gate xác định CỤM/chủ sở hữu —
    việc tính đoạn vi phạm thật (segment) vẫn do match_segments() ở matcher.py đảm nhiệm
    (Phase 03), không trùng lặp logic.
    """
    by_cluster: dict[str, list[dict]] = {}
    for c in candidates:
        by_cluster.setdefault(c["content_cluster_id"], []).append(c)

    min_frames_required = max(1, round(settings.FINGERPRINT_MIN_MATCH_SECONDS * settings.FINGERPRINT_FPS))

    # coverage = số QUERY FRAME riêng biệt khớp (không phải số candidate — 1 frame có thể
    # khớp nhiều kết quả top_k cùng 1 cụm, chỉ tính 1 lần).
    cluster_summaries = []
    for members in by_cluster.values():
        coverage = len({m["query_index"] for m in members})
        if coverage < min_frames_required:
            continue
        representative = min(members, key=lambda m: (m["first_seen_at"], m["media_id"]))
        cluster_summaries.append((coverage, representative))

    if not cluster_summaries:
        return None

    # Ưu tiên coverage cao nhất; nếu HÒA coverage giữa 2+ cụm, tie-break xác định
    # (deterministic) bằng first_seen_at sớm nhất của đại diện mỗi cụm — giống hệt cách
    # nhánh ảnh tie-break. Trước đây dùng "if coverage > best_coverage" (strict greater)
    # giữ nguyên cụm đến trước khi hòa — phụ thuộc thứ tự duyệt dict/kết quả search Milvus,
    # KHÔNG xác định (có thể ra kết quả khác nhau giữa 2 lần chạy cùng input).
    _, best_representative = min(
        cluster_summaries,
        key=lambda entry: (-entry[0], entry[1]["first_seen_at"], entry[1]["media_id"]),
    )
    return best_representative
