
from collections import defaultdict

from loguru import logger


def match_segments(
    query_fingerprints: list[dict],
    search_results: list[dict],
    similarity_threshold: float,
    min_match_seconds: int,
    max_gap_seconds: int,
    fps: int,
    exclude_media_id: int | None = None,
    exclude_creator_id: str | None = None,
    uploader_creator_id: str | None = None,
) -> list[dict]:

    threshold = similarity_threshold
    min_seconds = min_match_seconds
    max_gap = max_gap_seconds

    matches_by_source = defaultdict(list)

    for result in search_results:
        if result["score"] < threshold:
            continue
        if exclude_media_id and result["media_id"] == exclude_media_id:
            continue
        if exclude_creator_id and result.get("creator_id") == exclude_creator_id:
            continue

        if uploader_creator_id and result.get("original_creator_id") == uploader_creator_id:
            continue

        if result.get("is_violation"):
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

    segments = []

    for source_media_id, matches in matches_by_source.items():

        matches.sort(key=lambda m: m["target_timestamp"])

        tolerance = 0.5 / max(fps, 1)
        merged = _merge_consecutive(matches, max_gap, tolerance)

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


def match_image_violation(
    search_results: list[dict],
    similarity_threshold: float,
    exclude_media_id: str | None = None,
    exclude_creator_id: str | None = None,
    uploader_creator_id: str | None = None,
) -> list[dict]:
    threshold = similarity_threshold

    violations = []
    seen_media_ids = set()

    for result in search_results:
        if result["score"] < threshold:
            continue
        if exclude_media_id and result["media_id"] == exclude_media_id:
            continue
        if exclude_creator_id and result.get("creator_id") == exclude_creator_id:
            continue
        if uploader_creator_id and result.get("original_creator_id") == uploader_creator_id:
            continue
        if result.get("is_violation"):
            continue
        if result["media_id"] in seen_media_ids:
            continue
        seen_media_ids.add(result["media_id"])

        violations.append({
            "source_media_id": result["media_id"],
            "start_time_target": 0.0,
            "end_time_target": 0.0,
            "start_time_source": result.get("timestamp", 0.0),
            "end_time_source": result.get("timestamp", 0.0),
            "similarity_score": round(result["score"], 4),
            "violation_type": "IMAGE",
        })

    logger.info(f"Matcher: {len(violations)} image violations found")
    return violations


def _merge_consecutive(matches: list[dict], max_gap: float, tolerance: float = 0.5) -> list[dict]:

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

        if gap <= max_gap + tolerance:

            current["end_target"] = matches[i]["target_timestamp"]
            current["end_source"] = matches[i]["source_timestamp"]
            current["scores"].append(matches[i]["score"])
        else:

            current["avg_score"] = sum(current["scores"]) / len(current["scores"])
            segments.append(current)

            current = {
                "start_target": matches[i]["target_timestamp"],
                "end_target": matches[i]["target_timestamp"],
                "start_source": matches[i]["source_timestamp"],
                "end_source": matches[i]["source_timestamp"],
                "scores": [matches[i]["score"]],
            }


    current["avg_score"] = sum(current["scores"]) / len(current["scores"])
    segments.append(current)

    return segments
