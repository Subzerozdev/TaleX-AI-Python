"""
Content Router — Endpoint phân tích nội dung video (auto-tagging).
"""

from fastapi import APIRouter, Request

from app.core.rate_limiter import limiter
from app.schemas.content import ContentAnalyzeRequest, ContentAnalyzeResponse
from app.services.content_service import analyze_content

router = APIRouter(prefix="/api/v1", tags=["Content"])


@router.post("/content/analyze", response_model=ContentAnalyzeResponse)
@limiter.limit("20/minute")
def analyze(request: Request, body: ContentAnalyzeRequest):
    """
    Phân tích nội dung video → gợi ý tags, mood, age_rating.

    - Rate limit: 20 request/phút per IP.
    """
    return analyze_content(body)
