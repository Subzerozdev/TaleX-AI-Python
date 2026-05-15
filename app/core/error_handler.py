"""
Global Error Handler — Catch mọi exception, trả error format thống nhất.

Giống @ControllerAdvice + @ExceptionHandler trong Spring Boot.
Đảm bảo app KHÔNG BAO GIỜ trả stack trace cho client.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger


def register_error_handlers(app: FastAPI) -> None:
    """Đăng ký global error handlers vào FastAPI app."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Input không hợp lệ về mặt logic (đã qua Pydantic nhưng vẫn sai)."""
        logger.warning(f"ValueError: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_INPUT",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        """Lỗi runtime (model chưa load, ChromaDB chưa init...)."""
        logger.error(f"RuntimeError: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "AI Service chưa sẵn sàng, vui lòng thử lại sau.",
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Catch-all: mọi lỗi không mong đợi."""
        logger.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Đã xảy ra lỗi, vui lòng thử lại sau.",
                }
            },
        )
