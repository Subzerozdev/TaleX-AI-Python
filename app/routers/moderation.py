"""
Moderation Router — Endpoint kiểm duyệt nội dung video.
"""

from fastapi import APIRouter, Request

from app.core.rate_limiter import limiter
from app.schemas.moderation import ModerationRequest, ModerationResponse
from app.services.moderation_service import check_moderation

router = APIRouter(prefix="/api/v1", tags=["Moderation"])


@router.post("/moderation/check", response_model=ModerationResponse)
@limiter.limit("20/minute")
def moderate(request: Request, body: ModerationRequest):
    """
    Kiểm duyệt nội dung video → safe/unsafe + lý do.

    - Rate limit: 20 request/phút per IP.
    - Nếu AI lỗi → mặc định unsafe (an toàn hơn).
    """
    return check_moderation(body)
