"""
Extractor — Trích xuất frames từ video và đọc ảnh.

Video: FFmpeg đọc file → cắt 1 frame mỗi giây → trả list PIL Image.
Ảnh: Pillow đọc file → trả 1 PIL Image.

Tạm thời đọc file upload trực tiếp.
Sau này đổi sang S3 URL → chỉ sửa file này.
"""

import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from loguru import logger
from PIL import Image

from app.core.config import settings


def extract_frames_from_video(file_bytes: bytes) -> list[dict]:
    """
    Trích xuất frames từ video.

    Args:
        file_bytes: Nội dung file video (bytes).

    Returns:
        List of { "timestamp": float, "image": PIL.Image }
        Mỗi item = 1 frame tại 1 giây cụ thể.

    Ví dụ: video 10 giây → 10 items, timestamp 0.0 → 9.0
    """
    frames = []

    # Lưu bytes vào file tạm (FFmpeg cần đọc từ file)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Lấy duration của video
        duration = _get_video_duration(tmp_path)
        if duration <= 0:
            raise ValueError("Không thể xác định độ dài video.")

        logger.info(f"Extractor: video duration={duration:.1f}s, fps={settings.FINGERPRINT_FPS}")

        # Trích xuất frames bằng FFmpeg
        # -vf fps=1: lấy 1 frame mỗi giây
        # -f image2pipe: output ra pipe (stdout) dạng ảnh
        # -vcodec png: format PNG
        cmd = [
            "ffmpeg",
            "-i", tmp_path,
            "-vf", f"fps={settings.FINGERPRINT_FPS}",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-loglevel", "error",
            "pipe:1",
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg error: {error_msg[:200]}")

        # Parse output: FFmpeg ghi nhiều PNG liên tiếp vào stdout
        raw_data = result.stdout
        frames = _parse_png_stream(raw_data, duration)

        logger.info(f"Extractor: extracted {len(frames)} frames")

    finally:
        # Xóa file tạm
        Path(tmp_path).unlink(missing_ok=True)

    return frames


def extract_image(file_bytes: bytes) -> Image.Image:
    """
    Đọc ảnh từ bytes.

    Args:
        file_bytes: Nội dung file ảnh (bytes).

    Returns:
        PIL.Image object.
    """
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    logger.info(f"Extractor: image size={image.size}")
    return image


def _get_video_duration(file_path: str) -> float:
    """Lấy duration (giây) của video bằng ffprobe."""
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


def _parse_png_stream(raw_data: bytes, duration: float) -> list[dict]:
    """
    Parse chuỗi PNG liên tiếp từ FFmpeg stdout.

    FFmpeg ghi nhiều file PNG nối nhau vào stdout.
    Mỗi PNG bắt đầu bằng PNG header: b'\\x89PNG'
    """
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
            timestamp = float(i) / settings.FINGERPRINT_FPS
            frames.append({"timestamp": timestamp, "image": image})
        except Exception:
            continue

    return frames


def is_ffmpeg_available() -> bool:
    """Kiểm tra FFmpeg có sẵn không — dùng cho health check."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False
