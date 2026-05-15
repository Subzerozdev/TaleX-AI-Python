"""
Matcher — Nối kết quả rời rạc từ Milvus thành segments.

Milvus trả: "giây 10 trùng, giây 11 trùng, giây 13 trùng..."
Matcher nối: "đoạn trùng từ giây 10 → giây 13"

Cấu hình:
  SIMILARITY_THRESHOLD: score tối thiểu để tính trùng (0.85)
  MIN_MATCH_SECONDS: đoạn trùng tối thiểu (5 giây) — lọc trùng ngẫu nhiên
  MAX_GAP_SECONDS: gap tối đa giữa 2 giây vẫn tính cùng đoạn (2 giây)
"""

from collections import defaultdict

from loguru import logger

from app.core.config import settings


def match_segments(
    query_fingerprints: list[dict],
    search_results: list[dict],
    exclude_media_id: int | None = None,
) -> list[dict]:
    """
    Nối kết quả search rời rạc thành segments vi phạm.

    Args:
        query_fingerprints: List of { "timestamp": float, "vector": list }
                            — fingerprints của video mới.
        search_results: List of { "query_index": int, "media_id": int,
                                   "timestamp": float, "score": float }
                        — kết quả từ Milvus search.
        exclude_media_id: Bỏ qua media_id này (dùng khi upsert — không so với chính mình).

    Returns:
        List of {
            "source_media_id": int,
            "start_time_target": float,
            "end_time_target": float,
            "start_time_source": float,
            "end_time_source": float,
            "similarity_score": float,
            "violation_type": "VIDEO"
        }
    """
    threshold = settings.FINGERPRINT_SIMILARITY_THRESHOLD
    min_seconds = settings.FINGERPRINT_MIN_MATCH_SECONDS
    max_gap = settings.FINGERPRINT_MAX_GAP_SECONDS

    # Bước 1: Lọc matches theo threshold + exclude
    matches_by_source = defaultdict(list)

    for result in search_results:
        if result["score"] < threshold:
            continue
        if exclude_media_id and result["media_id"] == exclude_media_id:
            continue

        query_idx = result["query_index"]
        if query_idx >= len(query_fingerprints):
            continue

        target_ts = query_fingerprints[query_idx]["timestamp"]

        matches_by_source[result["media_id"]].append({
            "target_timestamp": target_ts,
            "source_timestamp": result["timestamp"],
            "score": result["score"],
        })

    # Bước 2: Với mỗi source_media_id, nối giây liên tiếp thành segments
    segments = []

    for source_media_id, matches in matches_by_source.items():
        # Sắp xếp theo target_timestamp
        matches.sort(key=lambda m: m["target_timestamp"])

        merged = _merge_consecutive(matches, max_gap)

        for seg in merged:
            duration = seg["end_target"] - seg["start_target"]
            if duration < min_seconds:
                continue  # Bỏ qua đoạn trùng quá ngắn

            segments.append({
                "source_media_id": source_media_id,
                "start_time_target": seg["start_target"],
                "end_time_target": seg["end_target"],
                "start_time_source": seg["start_source"],
                "end_time_source": seg["end_source"],
                "similarity_score": round(seg["avg_score"], 4),
                "violation_type": "VIDEO",
            })

    logger.info(f"Matcher: {len(segments)} violation segments found")
    return segments


def _merge_consecutive(matches: list[dict], max_gap: float) -> list[dict]:
    """
    Nối các giây liên tiếp (cho phép gap) thành segments.

    Input:  [giây 10, giây 11, giây 13, giây 14, giây 20]
    Output: [segment(10→14), segment(20→20)]  (gap 2 giây cho phép)
    """
    if not matches:
        return []

    segments = []
    current = {
        "start_target": matches[0]["target_timestamp"],
        "end_target": matches[0]["target_timestamp"],
        "start_source": matches[0]["source_timestamp"],
        "end_source": matches[0]["source_timestamp"],
        "scores": [matches[0]["score"]],
    }

    for i in range(1, len(matches)):
        gap = matches[i]["target_timestamp"] - current["end_target"]

        if gap <= max_gap + 0.5:  # +0.5 cho dung sai float
            # Nối tiếp vào segment hiện tại
            current["end_target"] = matches[i]["target_timestamp"]
            current["end_source"] = matches[i]["source_timestamp"]
            current["scores"].append(matches[i]["score"])
        else:
            # Gap quá lớn → đóng segment cũ, bắt đầu segment mới
            current["avg_score"] = sum(current["scores"]) / len(current["scores"])
            segments.append(current)

            current = {
                "start_target": matches[i]["target_timestamp"],
                "end_target": matches[i]["target_timestamp"],
                "start_source": matches[i]["source_timestamp"],
                "end_source": matches[i]["source_timestamp"],
                "scores": [matches[i]["score"]],
            }

    # Đóng segment cuối
    current["avg_score"] = sum(current["scores"]) / len(current["scores"])
    segments.append(current)

    return segments
