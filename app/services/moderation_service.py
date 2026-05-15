"""
Moderation Service — Kiểm duyệt nội dung video bằng Gemini.

Creator upload video → Staff duyệt → AI hỗ trợ kiểm tra tự động.
Phát hiện: bạo lực quá mức, bản quyền, nội dung người lớn, spam.
"""

from loguru import logger

from app.llm.gemini_client import generate_json, is_configured
from app.llm.prompts import MODERATION_PROMPT
from app.schemas.moderation import ModerationRequest, ModerationResponse

# Fallback khi Gemini lỗi — mặc định unsafe để Staff review thủ công
_FALLBACK = ModerationResponse(
    is_safe=False,
    confidence=0.0,
    flags=["auto_check_failed"],
    reason="Không thể kiểm tra tự động, cần Staff review thủ công.",
    suggestion="Vui lòng review nội dung này thủ công.",
)


def check_moderation(request: ModerationRequest) -> ModerationResponse:
    """Kiểm duyệt nội dung video → trả safe/unsafe + lý do."""

    logger.info(f"Moderation check: title='{request.title[:50]}'")

    if not is_configured():
        logger.warning("Gemini not configured. Returning fallback (unsafe).")
        return _FALLBACK

    try:
        prompt = MODERATION_PROMPT.format(
            title=request.title,
            description=request.description,
            tags=", ".join(request.tags) if request.tags else "không có",
        )

        result = generate_json(prompt)

        return ModerationResponse(
            is_safe=result.get("is_safe", False),
            confidence=result.get("confidence", 0.0),
            flags=result.get("flags", []),
            reason=result.get("reason", "Không có thông tin"),
            suggestion=result.get("suggestion", "Cần Staff review"),
        )

    except Exception as e:
        logger.error(f"Moderation check failed: {e}")
        return _FALLBACK
