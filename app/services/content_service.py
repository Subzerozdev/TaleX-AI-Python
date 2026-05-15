"""
Content Service — Auto-tagging video bằng Gemini.

Creator upload video → Spring Boot gọi endpoint này → AI phân tích
title + description → gợi ý tags, mood, age_rating.
"""

from loguru import logger

from app.llm.gemini_client import generate_json, is_configured
from app.llm.prompts import CONTENT_ANALYZE_PROMPT
from app.schemas.content import ContentAnalyzeRequest, ContentAnalyzeResponse

# Fallback khi Gemini lỗi
_FALLBACK = ContentAnalyzeResponse(
    suggested_tags=[],
    mood="unknown",
    age_rating="unknown",
    content_type="unknown",
)


def analyze_content(request: ContentAnalyzeRequest) -> ContentAnalyzeResponse:
    """Phân tích nội dung video → gợi ý tags, mood, age_rating."""

    logger.info(f"Content analyze: title='{request.title[:50]}'")

    if not is_configured():
        logger.warning("Gemini not configured. Returning fallback.")
        return _FALLBACK

    try:
        prompt = CONTENT_ANALYZE_PROMPT.format(
            title=request.title,
            description=request.description,
        )

        result = generate_json(prompt)

        return ContentAnalyzeResponse(
            suggested_tags=result.get("suggested_tags", []),
            mood=result.get("mood", "unknown"),
            age_rating=result.get("age_rating", "unknown"),
            content_type=result.get("content_type", "unknown"),
        )

    except Exception as e:
        logger.error(f"Content analyze failed: {e}")
        return _FALLBACK
