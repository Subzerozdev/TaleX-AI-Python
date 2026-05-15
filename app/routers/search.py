"""
Search Router — Endpoint tìm kiếm video.
"""

from fastapi import APIRouter, Request

from app.core.rate_limiter import limiter
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import search_videos

router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search(request: Request, body: SearchRequest):
    """
    Tìm kiếm video theo ngữ nghĩa (semantic search).

    - Rate limit: 30 request/phút per IP.
    - Không dùng LLM → nhanh (~200ms), không tốn tiền.
    """
    return search_videos(body)
