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
    exclude_creator_id: str | None = None,
) -> list[dict]:
    """
    Nối kết quả search rời rạc thành segments vi phạm.

    Args:
        query_fingerprints: List of { "timestamp": float, "vector": list }
                            — fingerprints của video mới.
        search_results: List of { "query_index": int, "media_id": int,
                                   "creator_id": str, "timestamp": float, "score": float }
                        — kết quả từ Milvus search.
        exclude_media_id: Bỏ qua media_id này (dùng khi upsert — không so với chính mình).
        exclude_creator_id: Bỏ qua mọi match cùng creator này — không tự báo "vi phạm"
            nội dung của chính mình (vd. nhân vật lặp lại xuyên suốt 1 bộ truyện).

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
        if exclude_creator_id and result.get("creator_id") == exclude_creator_id:
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

        # Dung sai float-dust = nửa chu kỳ 1 frame (1/FPS), thay vì hằng số cố định — ở
        # FPS mặc định (1) ra đúng 0.5 giống trước, nhưng còn đúng nếu FPS đổi (chu kỳ dài
        # hơn/ngắn hơn 1s thì dung sai cũng co giãn theo).
        tolerance = 0.5 / max(settings.FINGERPRINT_FPS, 1)
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
    exclude_media_id: str | None = None,
    exclude_creator_id: str | None = None,
) -> list[dict]:
    """
    Tìm vi phạm cho ẢNH — khác video, ảnh chỉ có 1 điểm fingerprint duy nhất
    (timestamp=0.0), không có khái niệm "đoạn thời gian" để nối. Áp MIN_MATCH_SECONDS
    của match_segments() lên ảnh khiến duration luôn = 0, loại bỏ MỌI kết quả bất kể
    giống nhau đến đâu — đây là lý do ảnh trùng 100% vẫn lọt qua kiểm duyệt. So khớp
    trực tiếp theo threshold ở đây, không qua bước nối đoạn.

    Args:
        search_results: kết quả search_similar() — đã loại trừ cùng creator ở Milvus.
        exclude_media_id: bỏ qua match trùng chính media_id đang xử lý (upsert).
        exclude_creator_id: bỏ qua match cùng creator (phòng hờ, Milvus đã lọc rồi).

    Returns:
        List of segment dict giống match_segments(), violation_type="IMAGE".
    """
    threshold = settings.FINGERPRINT_SIMILARITY_THRESHOLD

    violations = []
    seen_media_ids = set()

    for result in search_results:
        if result["score"] < threshold:
            continue
        if exclude_media_id and result["media_id"] == exclude_media_id:
            continue
        if exclude_creator_id and result.get("creator_id") == exclude_creator_id:
            continue
        if result["media_id"] in seen_media_ids:
            continue  # search_results đã sort theo score — giữ match tốt nhất mỗi nguồn
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

        if gap <= max_gap + tolerance:  # tolerance = nửa chu kỳ frame, hấp thụ dung sai float
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
