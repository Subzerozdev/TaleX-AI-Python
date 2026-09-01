

from loguru import logger

from app.llm.gemini_client import generate_json, is_configured
from app.llm.prompts import MODERATION_PROMPT
from app.schemas.moderation import ModerationRequest, ModerationResponse

_FALLBACK = ModerationResponse(
    is_safe=False,
    confidence=0.0,
    flags=["auto_check_failed"],
    reason="Không thể kiểm tra tự động, cần Staff review thủ công.",
    suggestion="Vui lòng review nội dung này thủ công.",
)


def check_moderation(request: ModerationRequest) -> ModerationResponse:

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
