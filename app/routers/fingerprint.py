"""
Fingerprint Router — API endpoints cho Content ID.

3 endpoints:
  POST   /api/v1/fingerprint/process     — upload file → xử lý fingerprint
  GET    /api/v1/fingerprint/{media_id}  — xem fingerprint info
  DELETE /api/v1/fingerprint/{media_id}  — xóa fingerprint
"""

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.core.rate_limiter import limiter
from app.schemas.fingerprint import DeleteResponse, FingerprintInfo, FingerprintResponse
from app.services.fingerprint_service import (
    delete_fingerprint,
    get_fingerprint_info,
    process_fingerprint,
)

router = APIRouter(prefix="/api/v1", tags=["Fingerprint"])


@router.post("/fingerprint/process", response_model=FingerprintResponse)
@limiter.limit("5/minute")
async def process(
    request: Request,
    media_id: int = Form(gt=0, description="ID video/ảnh"),
    file: UploadFile = File(description="File video hoặc ảnh"),
):
    """
    Upload file video/ảnh → tạo fingerprint → kiểm tra trùng lặp.

    - Rate limit: 5 request/phút (xử lý nặng).
    - Hỗ trợ: mp4, avi, mkv, mov, webm, flv, png, jpg, jpeg, webp, bmp.
    - Giới hạn: 100MB.
    - Trả về: Content ID + danh sách đoạn vi phạm (nếu có).
    """
    file_bytes = await file.read()
    filename = file.filename or "unknown"

    return process_fingerprint(media_id, file_bytes, filename)


@router.get("/fingerprint/{media_id}", response_model=FingerprintInfo)
def get_info(media_id: int):
    """Xem thông tin fingerprint đã lưu."""
    return get_fingerprint_info(media_id)


@router.delete("/fingerprint/{media_id}", response_model=DeleteResponse)
def delete(media_id: int):
    """Xóa tất cả fingerprints của 1 video/ảnh."""
    return delete_fingerprint(media_id)
