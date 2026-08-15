"""Video/Image moderation via AWS Rekognition frame sampling.

For VIDEO: extract frames (1/2s, max 30) in a single ffmpeg pass, then call
DetectModerationLabels on all frames in parallel (thread pool — Rekognition
client is blocking/sync). For IMAGE: single DetectModerationLabels call.
Cost: ~$0.001/frame = ~$0.03 per video (30 frames max).
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from loguru import logger
from PIL import Image
from app.aws.rekognition_client import detect_moderation_labels
from app.core.config import settings

# Cố tình hạ concurrency xuống 2 và thêm sleep để tránh lỗi ProvisionedThroughputExceededException
_REKOGNITION_MAX_CONCURRENCY = 2

# Nhóm L1 taxonomy dùng ngưỡng confidence riêng (thấp hơn) — xem giải thích ở
# REKOGNITION_VIOLENCE_CONFIDENCE_THRESHOLD trong config.py.
_LOWER_THRESHOLD_CATEGORIES = {"Violence", "Visually Disturbing"}


def _threshold_for_label(label: dict) -> float:
    # parent_name rỗng khi label CHÍNH LÀ danh mục gốc (L1, vd trả thẳng "Violence" không
    # qua label con) — phải tự kiểm cả label["name"] trong trường hợp đó, không chỉ parent_name.
    if label.get("parent_name") in _LOWER_THRESHOLD_CATEGORIES or label.get("name") in _LOWER_THRESHOLD_CATEGORIES:
        return settings.REKOGNITION_VIOLENCE_CONFIDENCE_THRESHOLD
    return settings.REKOGNITION_CONFIDENCE_THRESHOLD


def moderate_media(file_bytes: bytes, media_type: str, media_id: str, correlation_id: str) -> dict:
    """Run content moderation. Returns camelCase dict for Kafka."""
    try:
        if media_type == "IMAGE":
            violations, raw_responses = _moderate_image(file_bytes)
        else:
            violations, raw_responses = _moderate_video(file_bytes)

        is_safe = len(violations) == 0
        primary_label = None
        max_confidence = 0.0
        if violations:
            top = max(violations, key=lambda v: v["confidence"])
            primary_label = top["label"]
            max_confidence = top["confidence"]

        return {
            "mediaId": media_id,
            "correlationId": correlation_id,
            "isSafe": is_safe,
            "primaryLabel": primary_label,
            "confidenceScore": max_confidence,
            "violations": violations,
            "rawResponse": json.dumps(raw_responses, default=str),
            "processedAt": datetime.utcnow().isoformat(),
            "success": True,
            "errorMessage": None,
        }
    except Exception as e:
        logger.error(f"Moderation failed for mediaId={media_id}: {e}")
        return {
            "mediaId": media_id,
            "correlationId": correlation_id,
            "isSafe": False,
            "primaryLabel": None,
            "confidenceScore": 0.0,
            "violations": [],
            "rawResponse": "",
            "processedAt": datetime.utcnow().isoformat(),
            "success": False,
            "errorMessage": str(e),
        }


def _normalize_for_rekognition(file_bytes: bytes) -> bytes:
    """Re-encode upload as baseline RGB JPEG before calling Rekognition.

    Rekognition only accepts JPEG/PNG and rejects WEBP/BMP/CMYK-JPEG with
    InvalidImageFormatException — but the upload whitelist (IMAGE_EXTENSIONS
    in fingerprint_service.py) allows WEBP/BMP/JFIF for fingerprinting, which
    Pillow reads fine regardless of format. Re-encoding here closes that gap
    without restricting what creators can upload.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Could not normalize image for Rekognition, using original bytes: {e}")
        return file_bytes


def _moderate_image(file_bytes: bytes) -> tuple[list[dict], list]:
    """Single Rekognition call for image."""
    labels = detect_moderation_labels(_normalize_for_rekognition(file_bytes))
    violations = []
    for label in labels:
        if label["confidence"] >= _threshold_for_label(label):
            violations.append({
                "timestampMs": 0.0,
                "endTimestampMs": 0.0,
                "label": label["name"],
                "parentLabel": label["parent_name"],
                "confidence": label["confidence"],
                "suggestion": f"Image contains {label['name']} ({label['parent_name']})",
            })
    return violations, labels


def _moderate_video(file_bytes: bytes) -> tuple[list[dict], list]:
    """Extract frames, call Rekognition on each IN PARALLEL, aggregate results."""
    frames = _extract_moderation_frames(file_bytes)
    logger.info(f"Extracted {len(frames)} frames for moderation")

    all_raw: list[dict | None] = [None] * len(frames)
    all_violations = []

    def _check_frame(index: int, timestamp_sec: float, frame_bytes: bytes):
        import time
        time.sleep(0.5)
        return index, timestamp_sec, detect_moderation_labels(frame_bytes)

    with ThreadPoolExecutor(max_workers=_REKOGNITION_MAX_CONCURRENCY) as executor:
        futures = [
            executor.submit(_check_frame, i, timestamp_sec, frame_bytes)
            for i, (timestamp_sec, frame_bytes) in enumerate(frames)
        ]
        for future in as_completed(futures):
            index, timestamp_sec, labels = future.result()
            all_raw[index] = {"timestamp": timestamp_sec, "labels": labels}
            for label in labels:
                if label["confidence"] >= _threshold_for_label(label):
                    all_violations.append({
                        "timestampMs": timestamp_sec * 1000,
                        "endTimestampMs": (timestamp_sec + settings.MODERATION_FRAME_INTERVAL) * 1000,
                        "label": label["name"],
                        "parentLabel": label["parent_name"],
                        "confidence": label["confidence"],
                        "suggestion": f"Content '{label['name']}' detected at {timestamp_sec:.1f}s",
                    })

    return all_violations, all_raw


def _extract_moderation_frames(video_bytes: bytes) -> list[tuple[float, bytes]]:
    """Extract up to REKOGNITION_MAX_FRAMES frames in a SINGLE ffmpeg pass
    (previously spawned 1 ffmpeg process per frame — 30x process/seek overhead)."""
    base_interval = settings.MODERATION_FRAME_INTERVAL
    max_frames = settings.REKOGNITION_MAX_FRAMES

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frame_dir = tempfile.mkdtemp()
    try:
        duration = _get_video_duration(tmp_path)
        if duration <= 0:
            return []

        # Video dài hơn base_interval * max_frames -> dãn khoảng cách ra để rải đều
        # frame khắp video (giữ đúng hành vi cũ), thay vì chỉ lấy được đoạn đầu.
        effective_interval = max(base_interval, duration / max_frames)

        extract_result = subprocess.run(
            ["ffmpeg", "-i", tmp_path, "-vf", f"fps=1/{effective_interval}",
             "-frames:v", str(max_frames), "-f", "image2", "-c:v", "mjpeg",
             os.path.join(frame_dir, "frame_%03d.jpg")],
            capture_output=True, timeout=60,
        )
        if extract_result.returncode != 0:
            logger.warning(
                f"ffmpeg frame extraction failed: "
                f"{extract_result.stderr.decode(errors='ignore')[:500]}"
            )
            return []

        frames = []
        for i, filename in enumerate(sorted(os.listdir(frame_dir))[:max_frames]):
            with open(os.path.join(frame_dir, filename), "rb") as f:
                frames.append((round(i * effective_interval, 2), f.read()))

        return frames
    finally:
        os.unlink(tmp_path)
        shutil.rmtree(frame_dir, ignore_errors=True)


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
