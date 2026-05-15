"""
Chat Service — Xử lý logic chatbot (RAG pipeline đầy đủ).

Luồng:
  1. ChromaDB tìm video liên quan (Retrieval)
  2. Ghép prompt với context (Augmented)
  3. Gemini sinh reply (Generation)
  4. Parse response → trả về

Fallback: Gemini lỗi → trả kết quả ChromaDB + reply cố định.
"""

from loguru import logger

from app.core.sanitize import clean_text
from app.llm.gemini_client import generate_json, is_configured
from app.llm.prompts import CHAT_SYSTEM_PROMPT, build_chat_prompt
from app.rag.retriever import retrieve_videos
from app.schemas.chat import ChatRequest, ChatResponse

# Reply cố định khi Gemini lỗi
_FALLBACK_REPLY = "Xin lỗi, mình đang gặp sự cố kỹ thuật. Đây là một số video có thể phù hợp với bạn:"

# Reply khi off-topic
_OFFTOPIC_REPLY = "Mình là trợ lý của TaleX nên chỉ hỗ trợ về video truyện tranh và hoạt hình thôi nha! Bạn muốn mình gợi ý bộ truyện nào không?"


def chat(request: ChatRequest) -> ChatResponse:
    """
    Xử lý 1 câu hỏi chatbot.

    - Nếu Gemini chưa cấu hình → fallback (trả ChromaDB results + reply cố định).
    - Nếu Gemini lỗi/timeout → fallback.
    - Nếu câu hỏi off-topic → Gemini từ chối lịch sự.
    """
    # Sanitize input
    message = clean_text(request.message)
    logger.info(f"Chat: message='{message[:50]}...', has_preferences={request.preferences is not None}")

    # Bước 1 (R — Retrieval): Tìm video liên quan trong ChromaDB
    retrieval = _retrieve(message)
    video_ids = retrieval["video_ids"]
    videos_context = retrieval["context"]

    # Nếu Gemini chưa cấu hình → fallback ngay
    if not is_configured():
        logger.warning("Gemini not configured. Using fallback.")
        return ChatResponse(reply=_FALLBACK_REPLY, video_ids=video_ids, is_relevant=True)

    # Bước 2+3 (A+G — Augmented + Generation): Gọi Gemini
    try:
        result = _call_gemini(message, request, videos_context)
        return result

    except Exception as e:
        # Gemini lỗi → fallback: trả ChromaDB results + reply cố định
        logger.error(f"Gemini failed, using fallback: {e}")
        return ChatResponse(reply=_FALLBACK_REPLY, video_ids=video_ids, is_relevant=True)


def _retrieve(message: str) -> dict:
    """Bước R: Tìm video liên quan từ ChromaDB."""
    try:
        results = retrieve_videos(query=message, top_k=20)

        # Ghép thành text context cho prompt
        context_lines = []
        for vid, score, doc in zip(results["video_ids"], results["scores"], results["documents"]):
            context_lines.append(f"- video_id={vid} (score={score}): {doc[:150]}")

        context = "\n".join(context_lines) if context_lines else "Không tìm thấy video nào."

        return {
            "video_ids": results["video_ids"][:10],  # chỉ trả top 10 cho frontend
            "context": context,
        }

    except Exception as e:
        logger.error(f"ChromaDB retrieval failed: {e}")
        return {"video_ids": [], "context": "Không tìm thấy video nào."}


def _call_gemini(message: str, request: ChatRequest, videos_context: str) -> ChatResponse:
    """Bước A+G: Ghép prompt → gọi Gemini → parse response."""

    # Chuyển history từ Pydantic model sang dict
    history_dicts = None
    if request.history:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # Build prompt đầy đủ
    prompt = build_chat_prompt(
        message=message,
        videos_context=videos_context,
        preferences=request.preferences,
        history=history_dicts,
    )

    # Gọi Gemini
    result = generate_json(prompt, system_prompt=CHAT_SYSTEM_PROMPT)

    # Parse response
    is_relevant = result.get("is_relevant", True)
    reply = result.get("reply", _OFFTOPIC_REPLY)
    raw_ids = result.get("video_ids", [])

    # Đảm bảo video_ids là list[int]
    video_ids = []
    for vid in raw_ids:
        try:
            video_ids.append(int(vid))
        except (ValueError, TypeError):
            continue

    if not is_relevant:
        return ChatResponse(reply=reply, video_ids=[], is_relevant=False)

    return ChatResponse(reply=reply, video_ids=video_ids, is_relevant=True)
