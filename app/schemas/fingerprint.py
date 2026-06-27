"""
Fingerprint DTO — Request/Response schema cho Content ID.
"""

from pydantic import BaseModel, Field


class ViolationSegment(BaseModel):
    """1 đoạn vi phạm bản quyền."""

    source_media_id: str = Field(description="ID video gốc bị copy")
    start_time_target: float = Field(description="Giây bắt đầu đoạn trùng trong video mới")
    end_time_target: float = Field(description="Giây kết thúc đoạn trùng trong video mới")
    start_time_source: float = Field(description="Giây bắt đầu tương ứng trong video gốc")
    end_time_source: float = Field(description="Giây kết thúc tương ứng trong video gốc")
    similarity_score: float = Field(description="Độ trùng khớp (0.0 - 1.0)")
    violation_type: str = Field(description="VIDEO hoặc IMAGE")


class FingerprintResponse(BaseModel):
    """Kết quả sau khi xử lý fingerprint."""

    media_id: str
    content_id: str = Field(description="Mã định danh duy nhất, ví dụ: CID-000001")
    is_duplicate: bool = Field(description="True nếu phát hiện trùng")
    overall_similarity: float = Field(description="Độ trùng cao nhất (0.0 - 1.0)")
    fingerprint_count: int = Field(description="Số vectors đã tạo")
    violations: list[ViolationSegment] = Field(description="Danh sách đoạn vi phạm")


class FingerprintInfo(BaseModel):
    """Thông tin fingerprint đã lưu."""

    media_id: str
    content_id: str
    fingerprint_count: int
    is_stored: bool


class DeleteResponse(BaseModel):
    """Kết quả xóa fingerprint."""

    success: bool
    media_id: str
    deleted_count: int
    message: str
