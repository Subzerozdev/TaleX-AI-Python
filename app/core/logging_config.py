"""
Logging Configuration — Cấu hình loguru.

Ghi log ra console + file. File tự rotate khi quá 10MB.
"""

import sys

from loguru import logger


def setup_logging() -> None:
    """Cấu hình logging cho app."""

    # Xóa default handler
    logger.remove()

    # Console: hiện INFO trở lên
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # File: ghi DEBUG trở lên, rotate 10MB, giữ 5 file cũ
    logger.add(
        "logs/talex_ai.log",
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}",
    )
