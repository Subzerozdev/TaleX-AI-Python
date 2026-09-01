

import time
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.fingerprint.milvus_store import query_cluster_rows, search_similar


@dataclass
class ClusterResolution:
    matched: bool
    cluster_id: str
    original_creator_id: str
    first_seen_at: float

def resolve_content_cluster(
    vectors: list[bytes],
    creator_id: str,
    is_video: bool,
    cluster_threshold: float = settings.FINGERPRINT_CLUSTER_THRESHOLD,
    image_top_k: int = settings.FINGERPRINT_IMAGE_TOP_K,
    video_top_k: int = settings.FINGERPRINT_VIDEO_TOP_K,
    min_match_seconds: int = settings.FINGERPRINT_MIN_MATCH_SECONDS,
    fps: int = settings.FINGERPRINT_FPS,
) -> ClusterResolution:
    if not vectors:
        return _mint_new_cluster(creator_id)
    top_k = image_top_k if not is_video else video_top_k
    raw_matches = search_similar(vectors, top_k=top_k, exclude_creator_id=None)

    candidates = [
        m
        for m in raw_matches
        if m["score"] >= cluster_threshold
        and not m["is_violation"]
        and m["content_cluster_id"]
    ]

    if not candidates:
        return _mint_new_cluster(creator_id)

    if is_video:
        chosen = _pick_cluster_by_coverage(candidates, min_match_seconds, fps)
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


def _pick_cluster_by_coverage(
    candidates: list[dict], min_match_seconds: int, fps: int
) -> dict | None:
    by_cluster: dict[str, list[dict]] = {}
    for c in candidates:
        by_cluster.setdefault(c["content_cluster_id"], []).append(c)

    min_frames_required = max(1, round(min_match_seconds * fps))
    cluster_summaries = []
    for members in by_cluster.values():
        coverage = len({m["query_index"] for m in members})
        if coverage < min_frames_required:
            continue
        representative = min(members, key=lambda m: (m["first_seen_at"], m["media_id"]))
        cluster_summaries.append((coverage, representative))

    if not cluster_summaries:
        return None
    _, best_representative = min(
        cluster_summaries,
        key=lambda entry: (-entry[0], entry[1]["first_seen_at"], entry[1]["media_id"]),
    )
    return best_representative
