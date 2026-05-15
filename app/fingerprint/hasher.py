"""
Hasher — Tạo perceptual hash (vân tay) từ ảnh.

Flow: PIL Image → resize + grayscale → pHash → vector float.

pHash đặc biệt ở chỗ: ảnh bị crop, resize, thêm filter nhẹ
→ pHash vẫn tương tự nhau. Chỉ ảnh hoàn toàn khác mới cho pHash khác.
"""

import imagehash
import numpy as np
from loguru import logger
from PIL import Image

# pHash tạo ra hash 64-bit → vector 64 dimensions
HASH_SIZE = 8  # 8x8 = 64 bits
VECTOR_DIM = HASH_SIZE * HASH_SIZE  # 64


def hash_frame(image: Image.Image) -> list[float]:
    """
    Tạo pHash vector từ 1 ảnh.

    Args:
        image: PIL Image (bất kỳ size, bất kỳ mode).

    Returns:
        Vector 64 float (0.0 hoặc 1.0), đại diện vân tay ảnh.

    Bên trong pHash tự động:
      1. Resize ảnh về 32x32
      2. Chuyển grayscale
      3. Áp dụng DCT (Discrete Cosine Transform)
      4. Lấy 64 bit quan trọng nhất → hash
    """
    # imagehash.phash tự resize + grayscale bên trong
    phash = imagehash.phash(image, hash_size=HASH_SIZE)

    # Chuyển hash thành vector float [0.0, 1.0, 0.0, ...]
    vector = phash.hash.flatten().astype(np.float32).tolist()

    return vector


def hash_frames(frames: list[dict]) -> list[dict]:
    """
    Tạo pHash vectors cho nhiều frames cùng lúc.

    Args:
        frames: List of { "timestamp": float, "image": PIL.Image }

    Returns:
        List of { "timestamp": float, "vector": list[float] }
    """
    results = []

    for frame in frames:
        vector = hash_frame(frame["image"])
        results.append({
            "timestamp": frame["timestamp"],
            "vector": vector,
        })

    logger.info(f"Hasher: hashed {len(results)} frames → vectors (dim={VECTOR_DIM})")
    return results


def hash_image(image: Image.Image) -> list[float]:
    """
    Tạo pHash vector từ 1 ảnh (dùng cho IMAGE, không phải video frame).
    Logic giống hash_frame, tách ra để rõ ràng về mặt nghiệp vụ.
    """
    return hash_frame(image)
