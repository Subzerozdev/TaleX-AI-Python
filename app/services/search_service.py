"""
Search Service — Xử lý logic tìm kiếm video.

Tầng Service: nhận request đã validate → gọi retriever → trả kết quả.
KHÔNG biết HTTP request/response là gì (tách biệt với Router).
"""

from loguru import logger

from app.core.sanitize import clean_text
from app.rag.retriever import retrieve_videos
from app.schemas.search import SearchRequest, SearchResponse


def search_videos(request: SearchRequest) -> SearchResponse:
    """
    Tìm video theo query text.

    Luồng:
      1. Sanitize + validate query
      2. Gọi retriever (embedding + ChromaDB)
      3. Trả về video_ids + scores
    """
    query = clean_text(request.query)
    logger.info(f"Search: query='{query}', top_k={request.top_k}")

    try:
        results = retrieve_videos(query=query, top_k=request.top_k)

        response = SearchResponse(
            video_ids=results["video_ids"],
            scores=results["scores"],
        )

        logger.info(f"Search: found {len(response.video_ids)} videos")
        return response

    except Exception as e:
        # ChromaDB lỗi → trả rỗng thay vì crash
        logger.error(f"Search failed: {e}")
        return SearchResponse(video_ids=[], scores=[])
