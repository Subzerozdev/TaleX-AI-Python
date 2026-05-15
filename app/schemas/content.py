"""
Content Analyze DTO — Request/Response schema cho auto-tagging.
"""

from pydantic import BaseModel, Field


class ContentAnalyzeRequest(BaseModel):
    """Dữ liệu video cần phân tích."""

    title: str = Field(
        min_length=1,
        max_length=500,
        description="Tiêu đề video",
        examples=["Dark Knight Ep 5 - Cuộc chiến cuối cùng"],
    )
    description: str = Field(
        min_length=1,
        max_length=5000,
        description="Mô tả nội dung video",
    )


class ContentAnalyzeResponse(BaseModel):
    """Kết quả phân tích nội dung."""

    suggested_tags: list[str] = Field(description="Danh sách tags gợi ý")
    mood: str = Field(description="Tông cảm xúc (intense, wholesome, dark, comedic...)")
    age_rating: str = Field(description="Phân loại độ tuổi (7+, 13+, 16+, 18+)")
    content_type: str = Field(description="Loại nội dung (anime_series, manga_video...)")
