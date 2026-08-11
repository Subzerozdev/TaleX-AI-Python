from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from app.services.watermark_service import (
    embed_image_watermark,
    extract_image_watermark,
    embed_video_audio_watermark,
    extract_video_audio_watermark
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
            extracted_ids = extract_image_watermark(file_bytes)
            return extracted_ids
            
        elif media_type == "VIDEO":
            extracted_ids = extract_video_audio_watermark(file_bytes)
            return {
                "creator_id": f"ID: {extracted_ids['creator_id']} - Website: talex.pro.vn" if extracted_ids.get('creator_id') else None,
                "viewer_id": extracted_ids.get('viewer_id')
            }
            

        else:
            raise HTTPException(status_code=400, detail="Invalid media_type. Must be IMAGE or VIDEO")
            
    except ValueError as ve:
        return {"message": str(ve)}
    except Exception as e:
        logger.error(f"Error in extract_watermark_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))
