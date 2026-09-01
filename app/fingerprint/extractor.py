"""
Extractor — Trích xuất frames từ video và đọc ảnh.
Video: FFmpeg đọc file → cắt 1 frame mỗi giây → trả list PIL Image.
Ảnh: Pillow đọc file → trả 1 PIL Image.
"""

import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from loguru import logger
from PIL import Image

from app.core.config import settings


def extract_frames_from_video(
    file_bytes: bytes,
    fps: int = settings.FINGERPRINT_FPS,
    max_frames: int = settings.FINGERPRINT_MAX_FRAMES,
) -> list[dict]:
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Lấy duration của video
        duration = _get_video_duration(tmp_path)
        if duration <= 0:
            raise ValueError("Unable to determine video duration..")
        effective_fps = fps
        if duration * effective_fps > max_frames:
            effective_fps = max_frames / duration

        logger.info(f"Extractor: video duration={duration:.1f}s, fps={effective_fps:.4f}")
        cmd = [
            "ffmpeg",
            "-i", tmp_path,
            "-vf", f"fps={effective_fps}",
            "-frames:v", str(max_frames),
            "-f", "image2pipe",
            "-vcodec", "png",
            "-loglevel", "error",
            "pipe:1",
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg error: {error_msg[:200]}")

        raw_data = result.stdout
        frames = _parse_png_stream(raw_data, effective_fps)

        logger.info(f"Extractor: extracted {len(frames)} frames")

    finally:
        # Xóa file tạm
        Path(tmp_path).unlink(missing_ok=True)

    return frames


def extract_image(file_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    logger.info(f"Extractor: image size={image.size}")
    return image


def _get_video_duration(file_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        return 0.0

    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _parse_png_stream(raw_data: bytes, fps: float) -> list[dict]:
    frames = []
    png_header = b"\x89PNG"

    # Tìm vị trí bắt đầu của mỗi PNG
    positions = []
    start = 0
    while True:
        pos = raw_data.find(png_header, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    # Cắt từng PNG và tạo PIL Image
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(raw_data)
        png_bytes = raw_data[pos:end]

        try:
            image = Image.open(BytesIO(png_bytes)).convert("RGB")
            timestamp = float(i) / fps
            frames.append({"timestamp": timestamp, "image": image})
        except Exception:
            continue

    return frames


def is_ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False
