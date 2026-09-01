
import imagehash
import numpy as np
from loguru import logger
from PIL import Image

HASH_SIZE = 8  # 8x8 = 64 bits
VECTOR_DIM = HASH_SIZE * HASH_SIZE  # 64


def hash_frame(image: Image.Image) -> bytes:
    phash = imagehash.phash(image, hash_size=HASH_SIZE)
    bits = phash.hash.flatten().astype(np.uint8)
    packed = np.packbits(bits).tobytes()

    return packed


def hash_frames(frames: list[dict]) -> list[dict]:
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
    return hash_frame(image)
