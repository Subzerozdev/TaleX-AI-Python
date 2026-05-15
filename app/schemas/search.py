"""
Search DTO — Request/Response schema.
Giống DTO class trong Spring Boot. Pydantic tự validate dữ liệu.
"""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Dữ liệu đầu vào cho tìm kiếm video."""

    query: str = Field(
        min_length=1,
        max_length=200,
        description="Câu hỏi tìm kiếm",
        examples=["truyện dark fantasy nhân vật mạnh"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Số kết quả muốn lấy (1-50)",
    )


class SearchResponse(BaseModel):
    """Dữ liệu trả về sau khi tìm kiếm."""

    video_ids: list[int] = Field(description="Danh sách video IDs phù hợp")
    scores: list[float] = Field(description="Điểm tương đồng (0.0-1.0, càng cao càng giống)")
