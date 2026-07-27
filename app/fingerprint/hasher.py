"""
Hasher — Tạo perceptual hash (vân tay) từ ảnh.

Flow: PIL Image → resize + grayscale → pHash → 64-bit đóng gói thành bytes.

pHash đặc biệt ở chỗ: ảnh bị crop, resize, thêm filter nhẹ
→ pHash vẫn tương tự nhau. Chỉ ảnh hoàn toàn khác mới cho pHash khác.

QUAN TRỌNG: pHash phải so sánh bằng khoảng cách Hamming (đếm số bit khác nhau),
không phải cosine similarity — nên vector ở đây đóng gói thành bytes (BINARY_VECTOR
trong Milvus) thay vì list float, để Milvus tính đúng Hamming distance.
"""

import imagehash
import numpy as np
from loguru import logger
from PIL import Image

# pHash tạo ra hash 64-bit → đóng gói thành 8 bytes (BINARY_VECTOR dim=64 trong Milvus)
HASH_SIZE = 8  # 8x8 = 64 bits
VECTOR_DIM = HASH_SIZE * HASH_SIZE  # 64


def hash_frame(image: Image.Image) -> bytes:
    """
    Tạo pHash vân tay từ 1 ảnh, đóng gói dạng bytes để so bằng Hamming distance.

    Args:
        image: PIL Image (bất kỳ size, bất kỳ mode).

    Returns:
        8 bytes (64 bit đã đóng gói), đại diện vân tay ảnh.

    Bên trong pHash tự động:
      1. Resize ảnh về 32x32
      2. Chuyển grayscale
      3. Áp dụng DCT (Discrete Cosine Transform)
      4. Lấy 64 bit quan trọng nhất → hash
    """
    # imagehash.phash tự resize + grayscale bên trong
    phash = imagehash.phash(image, hash_size=HASH_SIZE)

    # Đóng gói 64 bit (0/1) thành 8 bytes — đúng định dạng Milvus BINARY_VECTOR cần.
    bits = phash.hash.flatten().astype(np.uint8)
    packed = np.packbits(bits).tobytes()

    return packed


def hash_frames(frames: list[dict]) -> list[dict]:
    """
    Tạo pHash vectors cho nhiều frames cùng lúc.

    Args:
        frames: List of { "timestamp": float, "image": PIL.Image }

    Returns:
        List of { "timestamp": float, "vector": bytes }
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


def hash_image(image: Image.Image) -> bytes:
    """
    Tạo pHash vector từ 1 ảnh (dùng cho IMAGE, không phải video frame).
    Logic giống hash_frame, tách ra để rõ ràng về mặt nghiệp vụ.
    """
    return hash_frame(image)
