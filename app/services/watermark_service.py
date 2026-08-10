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

def _ensure_3_channels(image_bytes: bytes) -> bytes:
    """Xử lý ảnh grayscale (2D) hoặc RGBA (4D) về BGR (3D) để tránh lỗi tuple index out of range của blind_watermark."""
    import cv2
    import numpy as np
    
    # Đọc ảnh từ byte array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Không thể đọc định dạng ảnh này")
        
    # Nếu là ảnh Grayscale (chỉ có 2 chiều H, W) -> Chuyển sang BGR (3 chiều)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Lưu lại thành file byte PNG
    success, encoded_image = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Lỗi khi encode ảnh sang định dạng PNG")
        
    return encoded_image.tobytes()

def embed_image_watermark(image_bytes: bytes, creator_id: str) -> bytes:
    """Nhúng watermark ẩn vào hình ảnh (DWT-DCT-SVD)."""
    padded_id = _pad_id(creator_id)
    
    # Xử lý lỗi tuple index out of range cho ảnh đen trắng
    processed_image_bytes = _ensure_3_channels(image_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        
        tmp_in.write(processed_image_bytes)
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
    # Xử lý lỗi tuple index out of range cho ảnh đen trắng
    processed_image_bytes = _ensure_3_channels(image_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(processed_image_bytes)
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
                
        except FileNotFoundError:
            raise ValueError("Lỗi: Không tìm thấy phần mềm FFmpeg trên máy chủ. Vui lòng cài đặt FFmpeg và thêm vào biến môi trường PATH.")
        except Exception as e:
            logger.error(f"Lỗi khi nhúng audio watermark cho video: {e}")
            raise e
        finally:
            os.remove(tmp_video_in.name)
            os.remove(tmp_audio_wm.name)
            os.remove(tmp_video_out.name)
            
    return out_bytes

def _binary_to_string(binary_str: str) -> str:
    """Chuyển chuỗi bit sang string (ví dụ '01000001' -> 'A')."""
    chars = []
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def extract_video_audio_watermark(video_bytes: bytes) -> str:
    """
    Trích xuất ID từ âm thanh siêu âm 18kHz của Video bằng FFT.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_out:
             
        tmp_video_in.write(video_bytes)
        tmp_video_in.flush()
        
        try:
            # 1. Dùng FFmpeg để tách audio ra thành file WAV mono 44.1kHz
            cmd = [
                "ffmpeg",
                "-y",
                "-i", tmp_video_in.name,
                "-vn",            # No video
                "-ac", "1",       # Mono
                "-ar", "44100",   # Sample rate 44.1kHz
                "-acodec", "pcm_s16le", # 16-bit PCM
                "-loglevel", "error",
                tmp_audio_out.name
            ]
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            
            # 2. Đọc file âm thanh
            sample_rate, audio_data = wavfile.read(tmp_audio_out.name)
            
            if len(audio_data) == 0:
                raise ValueError("Không tìm thấy luồng âm thanh trong video")
                
            # Đảm bảo audio_data là mảng 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
                
            # 3. Phân tích OOK ở 18kHz (từng chunk 0.1s)
            bit_duration = 0.1
            chunk_size = int(sample_rate * bit_duration)
            freq_target = 18000.0
            
            num_chunks = len(audio_data) // chunk_size
            powers = []
            
            for i in range(num_chunks):
                chunk = audio_data[i*chunk_size : (i+1)*chunk_size]
                # Thực hiện Fast Fourier Transform
                fft_result = np.fft.rfft(chunk)
                freqs = np.fft.rfftfreq(chunk_size, 1.0/sample_rate)
                
                # Tìm index của tần số gần 18000Hz nhất
                idx_18k = np.argmin(np.abs(freqs - freq_target))
                
                # Tính năng lượng quanh dải 18kHz (cộng dồn vài bin xung quanh để bù trừ nhiễu)
                power = np.sum(np.abs(fft_result[max(0, idx_18k-2) : min(len(fft_result), idx_18k+3)])**2)
                powers.append(power)
                
            if not powers:
                raise ValueError("Video quá ngắn để phân tích")
                
            # 4. Xác định ngưỡng (threshold) để phân biệt bit 1 và bit 0
            # Dùng median của top 10% năng lượng làm tín hiệu '1', phần còn lại là '0'
            sorted_powers = np.sort(powers)
            top_10_percent = sorted_powers[int(len(sorted_powers)*0.9):]
            if len(top_10_percent) == 0:
                threshold = np.mean(powers) * 1.5
            else:
                threshold = np.median(top_10_percent) * 0.3 # 30% của peak
            
            binary_sequence = ""
            for p in powers:
                if p > threshold:
                    binary_sequence += "1"
                else:
                    binary_sequence += "0"
                    
            # 5. Tìm chuỗi header '10101010' để đồng bộ (sync)
            header = "10101010"
            header_idx = binary_sequence.find(header)
            
            if header_idx == -1:
                raise ValueError("Không tìm thấy tín hiệu watermark trong video (Không thấy header)")
                
            # Trích xuất dữ liệu sau header
            data_bits = binary_sequence[header_idx + len(header):]
            
            # 6. Dịch ngược thành string
            creator_id_extracted = _binary_to_string(data_bits)
            
            # Lọc bỏ các ký tự rác (chỉ lấy ASCII in được)
            clean_id = ''.join(c for c in creator_id_extracted if 32 <= ord(c) <= 126)
            
            return clean_id
            
            
        except FileNotFoundError:
            logger.error("Lỗi: Không tìm thấy FFmpeg trên máy chủ.")
            raise ValueError("Lỗi: Không tìm thấy phần mềm FFmpeg trên máy chủ. Vui lòng cài đặt FFmpeg và thêm vào biến môi trường PATH.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Lỗi FFmpeg khi trích xuất audio: {e}")
            raise RuntimeError("Lỗi khi tách âm thanh từ video")
        except Exception as e:
            logger.error(f"Lỗi khi giải mã audio watermark: {e}")
            raise e
        finally:
            os.remove(tmp_video_in.name)
            if os.path.exists(tmp_audio_out.name):
                os.remove(tmp_audio_out.name)
