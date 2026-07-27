"""Service for generating media previews (image blur, video cut)."""

import io
import os
import tempfile
import ffmpeg
from PIL import Image, ImageFilter
from loguru import logger

def generate_image_preview(file_bytes: bytes) -> bytes:
    """Apply Gaussian blur to an image for preview."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=20))
        out = io.BytesIO()
        blurred.save(out, format="JPEG", quality=85)
        logger.info("Successfully generated image preview")
        return out.getvalue()
    except Exception as e:
        logger.error(f"Failed to generate image preview: {e}")
        raise

def generate_video_preview(file_bytes: bytes) -> bytes:
    """Extract the first 10 seconds of a video for preview."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.mp4")
        out_path = os.path.join(tmpdir, "output.mp4")
        
        with open(in_path, "wb") as f:
            f.write(file_bytes)
        
        try:
            (
                ffmpeg
                .input(in_path, ss=0, t=10)
                .output(out_path, vcodec='libx264', acodec='aac', preset='fast', crf=28)
                .run(capture_stdout=True, capture_stderr=True)
            )
            with open(out_path, "rb") as f:
                data = f.read()
            logger.info("Successfully generated video preview")
            return data
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error generating video preview: {err_msg}")
            raise
