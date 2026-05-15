"""
Chat Router — Endpoint chatbot.

Spring Boot gọi endpoint này khi user gửi tin nhắn trong chatbox.
"""

from fastapi import APIRouter, Request

from app.core.rate_limiter import limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat_endpoint(request: Request, body: ChatRequest):
    """
    Chatbot AI — RAG pipeline đầy đủ.

    - Rate limit: 10 request/phút per IP.
    - Tìm video liên quan (ChromaDB).
    - Gemini sinh reply thân thiện + chọn video phù hợp.
    - Fallback khi Gemini lỗi.
    """
    return chat(body)
