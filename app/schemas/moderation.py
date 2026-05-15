"""
Moderation DTO — Request/Response schema cho kiểm duyệt nội dung.
"""

from pydantic import BaseModel, Field


class ModerationRequest(BaseModel):
    """Dữ liệu video cần kiểm duyệt."""

    title: str = Field(min_length=1, max_length=500, description="Tiêu đề video")
    description: str = Field(min_length=1, max_length=5000, description="Mô tả nội dung video")
    tags: list[str] = Field(default=[], max_length=20, description="Danh sách tags")


class ModerationResponse(BaseModel):
    """Kết quả kiểm duyệt."""

    is_safe: bool = Field(description="True nếu nội dung an toàn")
    confidence: float = Field(description="Độ tin cậy (0.0 - 1.0)")
    flags: list[str] = Field(description="Danh sách vi phạm phát hiện được")
    reason: str = Field(description="Giải thích ngắn gọn")
    suggestion: str = Field(description="Gợi ý cho Staff")
