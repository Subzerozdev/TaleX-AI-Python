"""
Chat DTO — Request/Response schema cho chatbot.
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """1 message trong lịch sử hội thoại."""

    role: str = Field(description="'user' hoặc 'assistant'", examples=["user"])
    content: str = Field(description="Nội dung message")


class ChatRequest(BaseModel):
    """Dữ liệu đầu vào cho chatbot."""

    message: str = Field(
        min_length=2,
        max_length=500,
        description="Câu hỏi của user",
        examples=["Tìm truyện giống Solo Leveling nhưng có tình cảm"],
    )
    preferences: dict | None = Field(
        default=None,
        description="Sở thích user (null nếu guest). Ví dụ: {favorite_genres: [...], liked_video_ids: [...]}",
    )
    history: list[ChatMessage] | None = Field(
        default=None,
        max_length=20,
        description="Lịch sử hội thoại (tối đa 20 messages gần nhất, null nếu câu đầu tiên)",
    )


class ChatResponse(BaseModel):
    """Dữ liệu trả về từ chatbot."""

    reply: str = Field(description="Câu trả lời thân thiện từ LLM")
    video_ids: list[int] = Field(description="Danh sách video IDs gợi ý")
    is_relevant: bool = Field(description="True nếu câu hỏi liên quan TaleX, False nếu off-topic")
