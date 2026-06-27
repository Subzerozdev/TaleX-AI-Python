"""Video/Image moderation via AWS Rekognition frame sampling.

For VIDEO: extract frames (1/2s, max 30), call DetectModerationLabels per frame.
For IMAGE: single DetectModerationLabels call.
Cost: ~$0.001/frame = ~$0.03 per video (30 frames max).
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime

from loguru import logger
from app.aws.rekognition_client import detect_moderation_labels
from app.core.config import settings


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


def _moderate_image(file_bytes: bytes) -> tuple[list[dict], list]:
    """Single Rekognition call for image."""
    labels = detect_moderation_labels(file_bytes)
    violations = []
    threshold = settings.REKOGNITION_CONFIDENCE_THRESHOLD
    for label in labels:
        if label["confidence"] >= threshold:
            violations.append({
                "timestampMs": 0.0,
                "endTimestampMs": 0.0,
                "label": label["name"],
                "confidence": label["confidence"],
                "suggestion": f"Image contains {label['name']} ({label['parent_name']})",
            })
    return violations, labels


def _moderate_video(file_bytes: bytes) -> tuple[list[dict], list]:
    """Extract frames, call Rekognition on each, aggregate results."""
    frames = _extract_moderation_frames(file_bytes)
    logger.info(f"Extracted {len(frames)} frames for moderation")

    all_violations = []
    all_raw = []
    threshold = settings.REKOGNITION_CONFIDENCE_THRESHOLD

    for timestamp_sec, frame_bytes in frames:
        labels = detect_moderation_labels(frame_bytes)
        all_raw.append({"timestamp": timestamp_sec, "labels": labels})
        for label in labels:
            if label["confidence"] >= threshold:
                all_violations.append({
                    "timestampMs": timestamp_sec * 1000,
                    "endTimestampMs": (timestamp_sec + settings.MODERATION_FRAME_INTERVAL) * 1000,
                    "label": label["name"],
                    "confidence": label["confidence"],
                    "suggestion": f"Content '{label['name']}' detected at {timestamp_sec:.1f}s",
                })

    return all_violations, all_raw


def _extract_moderation_frames(video_bytes: bytes) -> list[tuple[float, bytes]]:
    """Extract frames at interval, max REKOGNITION_MAX_FRAMES frames."""
    interval = settings.MODERATION_FRAME_INTERVAL
    max_frames = settings.REKOGNITION_MAX_FRAMES

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        duration = _get_video_duration(tmp_path)
        if duration <= 0:
            return []

        total_possible = int(duration / interval)
        step = max(1, total_possible // max_frames) if total_possible > max_frames else 1

        frames = []
        for i in range(0, total_possible, step):
            if len(frames) >= max_frames:
                break
            timestamp = i * interval
            frame_bytes = _extract_single_frame(tmp_path, timestamp)
            if frame_bytes:
                frames.append((timestamp, frame_bytes))

        return frames
    finally:
        os.unlink(tmp_path)


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


def _extract_single_frame(video_path: str, timestamp: float) -> bytes | None:
    """Extract a single JPEG frame at given timestamp."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
            capture_output=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return None
    except Exception:
        return None
