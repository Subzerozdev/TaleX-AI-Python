import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from app.services.watermark_service import (
    embed_image_watermark,
    extract_image_watermark,
    embed_video_audio_watermark,
    extract_ab_watermark_hls
)
from loguru import logger

router = APIRouter(
    prefix="/watermark",
    tags=["Watermark"]
)

@router.post("/embed")
async def embed_watermark_api(
    file: UploadFile = File(...),
    creator_id: str = Form(...),
    viewer_id: str = Form(""),
    media_type: str = Form(...)  # "IMAGE" or "VIDEO"
):
    """
    API phục vụ việc test nhúng watermark thủ công.
    Nhận vào ảnh hoặc video, trả về file đã được nhúng watermark.
    """
    try:
        file_bytes = await file.read()
        
        if media_type == "IMAGE":
            watermarked_bytes = embed_image_watermark(file_bytes, creator_id, viewer_id)
            return Response(content=watermarked_bytes, media_type="image/png")
            
        elif media_type == "VIDEO":
            watermarked_bytes = embed_video_audio_watermark(file_bytes, creator_id)
            return Response(content=watermarked_bytes, media_type="video/mp4")
            
        else:
            raise HTTPException(status_code=400, detail="Invalid media_type. Must be IMAGE or VIDEO")
            
    except Exception as e:
        logger.error(f"Error in embed_watermark_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract")
async def extract_watermark_api(
    file: UploadFile = File(...),
    media_type: str = Form(...)  # "IMAGE" or "VIDEO"
):
    """
    API phục vụ việc trích xuất watermark (tìm creator_id).
    """
    try:
        file_bytes = await file.read()
        
        if media_type == "IMAGE":
            # 1. Thử giải mã Blind Watermark (DWT-DCT)
            extracted_ids = await asyncio.to_thread(extract_image_watermark, file_bytes)
            
            # 2. Nếu thất bại (ảnh bị crop/sửa đổi), dùng AI Fingerprint (Milvus) làm lưới dự phòng
            if not extracted_ids.get("creator_id"):
                logger.info("Blind Watermark thất bại, kích hoạt Fallback AI Fingerprint...")
                try:
                    from app.fingerprint.extractor import extract_image
                    from app.fingerprint.hasher import hash_image
                    from app.fingerprint.content_ownership import resolve_content_cluster
                    
                    # Chạy nặng trên thread để không block API
                    def _fallback_search():
                        image = extract_image(file_bytes)
                        vector = hash_image(image)
                        cluster = resolve_content_cluster([vector], creator_id="", is_video=False)
                        if cluster.matched:
                            return cluster.original_creator_id
                        return None
                        
                    creator_id_fallback = await asyncio.to_thread(_fallback_search)
                    if creator_id_fallback:
                        extracted_ids["creator_id"] = creator_id_fallback
                        logger.info(f"Fallback thành công! Tìm thấy Creator ID = {creator_id_fallback}")
                except Exception as e:
                    logger.error(f"Lỗi khi chạy Fallback Fingerprint: {e}")
                    
            return extracted_ids
            
        elif media_type == "VIDEO":
            extracted_ids = extract_ab_watermark_hls(file_bytes)
            return {
                "creator_id": extracted_ids.get('creator_id'),
                "viewer_id": extracted_ids.get('viewer_id')
            }
            

        else:
            raise HTTPException(status_code=400, detail="Invalid media_type. Must be IMAGE or VIDEO")
            
    except ValueError as ve:
        return {"message": str(ve)}
    except Exception as e:
        logger.error(f"Error in extract_watermark_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))
