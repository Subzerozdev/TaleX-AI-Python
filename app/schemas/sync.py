"""
Sync DTO — Request/Response schema.
Dùng khi Spring Boot gửi video data để đồng bộ vào ChromaDB.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SyncRequest(BaseModel):
    """Dữ liệu video từ Spring Boot gửi sang."""

    video_id: int = Field(gt=0, description="ID video từ PostgreSQL")
    title: str = Field(
        min_length=1,
        max_length=500,
        description="Tiêu đề video",
        examples=["Solo Leveling Ep 1 - Thợ săn yếu nhất"],
    )
    description: str = Field(
        min_length=1,
        max_length=5000,
        description="Mô tả nội dung video",
    )
    tags: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Danh sách tags (1-20 tags)",
        examples=[["action", "fantasy", "zero-to-hero"]],
    )
    action: Literal["create", "update", "delete"] = Field(
        description="Hành động: create (thêm mới), update (cập nhật), delete (xóa)",
    )


class SyncResponse(BaseModel):
    """Kết quả sau khi sync."""

    success: bool
    video_id: int
    action: str
    message: str
