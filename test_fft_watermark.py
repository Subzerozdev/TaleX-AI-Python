import asyncio
import os
import tempfile
import subprocess
from app.services.watermark_service import embed_video_audio_watermark, extract_video_audio_watermark

def test_watermark():
    creator_id = "USER_12345"
    
    print("Tạo video giả lập (10s) bằng ffmpeg...")
    blank_video_path = "blank_test.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", 
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=10", 
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", 
            "-c:v", "libx264", "-c:a", "aac", "-shortest", blank_video_path
        ], check=True, capture_output=True)
    except Exception as e:
        print(f"Không thể chạy ffmpeg để tạo video (có thể ffmpeg chưa cài trên PATH): {e}")
        return

    print("Đọc video giả lập...")
    with open(blank_video_path, "rb") as f:
        video_bytes = f.read()
        
    print(f"Nhúng watermark với ID: {creator_id}...")
    watermarked_bytes = embed_video_audio_watermark(video_bytes, creator_id)
    
    print("Giải mã watermark từ video đã nhúng...")
    extracted_id = extract_video_audio_watermark(watermarked_bytes)
    
    print(f"Kết quả trích xuất: '{extracted_id}'")
    if extracted_id == creator_id:
        print("THÀNH CÔNG: ID trích xuất khớp với ID gốc!")
    else:
        print("THẤT BẠI: ID trích xuất KHÔNG khớp với ID gốc!")
        
    os.remove(blank_video_path)

if __name__ == "__main__":
    test_watermark()
