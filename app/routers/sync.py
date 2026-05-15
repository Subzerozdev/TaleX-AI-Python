"""
Sync Router — Endpoint nhận video data từ Spring Boot.

Spring Boot gọi endpoint này mỗi khi Creator upload/sửa/xóa video
để đồng bộ data vào ChromaDB cho AI tìm kiếm.
"""

from fastapi import APIRouter

from app.schemas.sync import SyncRequest, SyncResponse
from app.services.sync_service import sync_video

router = APIRouter(prefix="/api/v1", tags=["Sync"])


@router.post("/sync", response_model=SyncResponse)
def sync(request: SyncRequest):
    """
    Đồng bộ video vào ChromaDB.

    - **create**: Creator upload video mới → thêm vào ChromaDB.
    - **update**: Creator sửa video → cập nhật vector trong ChromaDB.
    - **delete**: Creator xóa video → xóa khỏi ChromaDB.
    """
    return sync_video(request)
