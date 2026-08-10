import os
import tempfile
import subprocess
import numpy as np
from scipy.io import wavfile
from loguru import logger
from app.core.config import settings
from blind_watermark import WaterMark

# ---------------------------------------------------------
# IMAGE BLIND WATERMARK
# ---------------------------------------------------------

WM_SHAPE_LENGTH = 64

def _pad_id(creator_id: str) -> str:
    """Đệm chuỗi cho đủ chiều dài cố định để thuật toán extract hoạt động đúng."""
    # Lọc bỏ các ký tự không phải ASCII để đảm bảo 1 character = 1 byte
    creator_id_ascii = creator_id.encode('ascii', 'ignore').decode('ascii')
    payload = f"ID: {creator_id_ascii} - Website: talex.pro.vn"
    # Pad bằng khoảng trắng để đạt chuẩn đúng 64 ký tự (64 bytes)
    return payload.ljust(WM_SHAPE_LENGTH, ' ')[:WM_SHAPE_LENGTH]

def _unpad_id(padded_id: str) -> str:
    # Dọn dẹp khoảng trắng dư thừa
    return padded_id.strip()

def embed_image_watermark(image_bytes: bytes, creator_id: str) -> bytes:
    """Nhúng watermark ẩn vào hình ảnh (DWT-DCT-SVD)."""
    padded_id = _pad_id(creator_id)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        
        tmp_in.write(image_bytes)
        tmp_in.flush()
        
        try:
            bwm = WaterMark(
                password_img=settings.WATERMARK_PASSWORD_IMG, 
                password_wm=settings.WATERMARK_PASSWORD_WM
            )
            bwm.read_img(tmp_in.name)
            # Sử dụng chuẩn mode='str' y hệt như bản gốc
            bwm.read_wm(padded_id, mode='str')
            bwm.embed(tmp_out.name)
            
            with open(tmp_out.name, "rb") as f:
                out_bytes = f.read()
        except Exception as e:
            logger.error(f"Lỗi khi nhúng blind watermark ảnh: {e}")
            raise e
        finally:
            os.remove(tmp_in.name)
            os.remove(tmp_out.name)
            
    return out_bytes

def extract_image_watermark(image_bytes: bytes) -> str:
    """Trích xuất watermark ẩn từ hình ảnh."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(image_bytes)
        tmp_in.flush()
        
        try:
            bwm = WaterMark(
                password_img=settings.WATERMARK_PASSWORD_IMG, 
                password_wm=settings.WATERMARK_PASSWORD_WM
            )
            # Vì chuỗi luôn bắt đầu bằng chữ "I" (mã ASCII là 73 = 01001001)
            # Hàm int.from_bytes của thư viện sẽ CẮT MẤT 1 số 0 ở đầu của bit đầu tiên.
            # Do đó 64 ký tự = 64 * 8 = 512 bits, trừ đi 1 bit số 0 bị cắt, kết quả LUÔN LÀ 511 bits!
            wm_shape_exact = (WM_SHAPE_LENGTH * 8) - 1
            extracted = bwm.extract(tmp_in.name, wm_shape=wm_shape_exact, mode='str')
            return _unpad_id(extracted)
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất blind watermark ảnh: {e}")
            raise e
        finally:
            os.remove(tmp_in.name)


# ---------------------------------------------------------
# VIDEO AUDIO WATERMARK
# ---------------------------------------------------------

def _string_to_binary(text: str) -> str:
    """Chuyển string sang chuỗi bit (ví dụ 'A' -> '01000001')."""
    return ''.join(format(ord(c), '08b') for c in text)

def _generate_ultrasound_wav(creator_id: str, output_wav_path: str):
    """
    Sinh ra file âm thanh WAV chứa sóng siêu âm mã hóa creator_id.
    Kỹ thuật: On-Off Keying (OOK) ở tần số 18kHz.
    """
    freq = 18000.0  # 18kHz (gần như ngoài ngưỡng nghe của người lớn)
    sample_rate = 44100
    bit_duration = 0.1  # 10 bits per second (mỗi bit kéo dài 0.1s)
    
    # Thêm header '10101010' để dễ nhận diện lúc decode
    binary_str = "10101010" + _string_to_binary(creator_id)
    
    t = np.linspace(0, bit_duration, int(sample_rate * bit_duration), endpoint=False)
    
    audio_data = []
    for bit in binary_str:
        if bit == '1':
            # Sóng sine 18kHz
            wave = np.sin(2 * np.pi * freq * t)
        else:
            # Im lặng
            wave = np.zeros_like(t)
        audio_data.append(wave)
        
    # Gộp tất cả bit lại thành 1 mảng 1D
    audio_signal = np.concatenate(audio_data)
    
    # Chuẩn hóa về định dạng int16 (-32768 đến 32767)
    # Để âm lượng siêu âm nhỏ (10% max volume) để không làm méo tiếng gốc quá nhiều
    audio_signal = np.int16(audio_signal * 3276 * 0.1)
    
    wavfile.write(output_wav_path, sample_rate, audio_signal)


def embed_video_audio_watermark(video_bytes: bytes, creator_id: str) -> bytes:
    """
    Trộn sóng siêu âm chứa ID vào luồng âm thanh của Video gốc.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_wm, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_out:
             
        tmp_video_in.write(video_bytes)
        tmp_video_in.flush()
        
        try:
            # 1. Sinh file WAV siêu âm chứa creator_id
            _generate_ultrasound_wav(creator_id, tmp_audio_wm.name)
            
            # 2. Dùng FFmpeg trộn 2 luồng audio lại, KHÔNG ĐỤNG CHẠM VÀO LUỒNG VIDEO (copy)
            # amix=inputs=2:duration=first -> Trộn 2 tiếng, độ dài bằng file video đầu vào
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", tmp_video_in.name,
                "-i", tmp_audio_wm.name,
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[a]",
                "-map", "0:v",  # Giữ nguyên video gốc
                "-map", "[a]",  # Lấy luồng audio đã mix
                "-c:v", "copy", # Copy nguyên xi video, siêu nhanh, không encode lại
                "-c:a", "aac",  # Encode lại audio
                "-loglevel", "error",
                tmp_video_out.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0:
                error_msg = result.stderr.decode("utf-8", errors="replace")
                logger.error(f"FFmpeg error: {error_msg}")
                raise RuntimeError(f"FFmpeg audio mixing failed: {error_msg[:200]}")
                
            with open(tmp_video_out.name, "rb") as f:
                out_bytes = f.read()
                
        except Exception as e:
            logger.error(f"Lỗi khi nhúng audio watermark cho video: {e}")
            raise e
        finally:
            os.remove(tmp_video_in.name)
            os.remove(tmp_audio_wm.name)
            os.remove(tmp_video_out.name)
            
    return out_bytes
