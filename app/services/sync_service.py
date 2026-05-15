"""
Sync Service — Đồng bộ video data từ Spring Boot vào ChromaDB.

Khi Creator upload/sửa/xóa video trên Spring Boot:
  Spring Boot gọi POST /api/v1/sync → AI Service nhận → cập nhật ChromaDB.
"""

from loguru import logger

from app.rag.embeddings import embed_text
from app.rag.vector_store import add_video, delete_video
from app.schemas.sync import SyncRequest, SyncResponse


def _build_document(request: SyncRequest) -> str:
    """Ghép title + description + tags thành 1 đoạn text để embedding."""
    return f"{request.title}. {request.description}. Tags: {', '.join(request.tags)}"


def sync_video(request: SyncRequest) -> SyncResponse:
    """
    Đồng bộ 1 video vào ChromaDB.

    3 actions:
      - create: thêm video mới (chuyển text → vector → lưu ChromaDB)
      - update: cập nhật video (upsert — ghi đè vector cũ)
      - delete: xóa video khỏi ChromaDB
    """
    logger.info(f"Sync: video_id={request.video_id}, action={request.action}")

    try:
        if request.action == "delete":
            delete_video(request.video_id)
            return SyncResponse(
                success=True,
                video_id=request.video_id,
                action="deleted",
                message=f"Video {request.video_id} đã xóa khỏi ChromaDB.",
            )

        # create hoặc update → cùng logic (upsert)
        document = _build_document(request)
        embedding = embed_text(document)

        add_video(
            video_id=request.video_id,
            document=document,
            embedding=embedding,
            metadata={"tags": ",".join(request.tags)},
        )

        action_label = "created" if request.action == "create" else "updated"
        return SyncResponse(
            success=True,
            video_id=request.video_id,
            action=action_label,
            message=f"Video {request.video_id} đã {action_label} trong ChromaDB.",
        )

    except Exception as e:
        logger.error(f"Sync failed for video {request.video_id}: {e}")
        return SyncResponse(
            success=False,
            video_id=request.video_id,
            action=request.action,
            message=f"Lỗi khi sync video {request.video_id}: {str(e)}",
        )
