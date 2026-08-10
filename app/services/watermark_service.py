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
    
    tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        
    tmp_in.write(processed_image_bytes)
    tmp_in.close()
    tmp_out.close()
        
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
        if os.path.exists(tmp_in.name): os.remove(tmp_in.name)
        if os.path.exists(tmp_out.name): os.remove(tmp_out.name)
            
    return out_bytes

def extract_image_watermark(image_bytes: bytes) -> str:
    """Trích xuất watermark ẩn từ hình ảnh."""
    # Xử lý lỗi tuple index out of range cho ảnh đen trắng
    processed_image_bytes = _ensure_3_channels(image_bytes)
    
    tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_in.write(processed_image_bytes)
    tmp_in.close()
        
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
        if os.path.exists(tmp_in.name): os.remove(tmp_in.name)


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
    # Tăng âm lượng siêu âm lên 50% max volume (32767 * 0.5) để sống sót qua AAC compression của AWS.
    # Âm thanh 18kHz ở 50% âm lượng vẫn rất khó nghe thấy đối với hầu hết người lớn.
    audio_signal = np.int16(audio_signal * 32767 * 0.5)
    
    wavfile.write(output_wav_path, sample_rate, audio_signal)


def embed_video_audio_watermark(video_bytes: bytes, creator_id: str) -> bytes:
    """
    Trộn sóng siêu âm chứa ID vào luồng âm thanh của Video gốc.
    """
    tmp_video_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_audio_wm = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_video_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
             
    tmp_video_in.write(video_bytes)
    
    # Đóng file handle trên Windows để FFmpeg không bị lỗi Permission (WinError 32)
    tmp_video_in.close()
    tmp_audio_wm.close()
    tmp_video_out.close()
        
    try:
        import json
        
        # 1. Kiểm tra xem video gốc có luồng audio hay không và lấy thời lượng
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", tmp_video_in.name
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = False
        video_duration = None
        if probe_res.stdout:
            try:
                info = json.loads(probe_res.stdout)
                has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
                video_duration = info.get("format", {}).get("duration")
            except json.JSONDecodeError:
                pass
                
        # 2. Sinh file WAV siêu âm chứa creator_id
        _generate_ultrasound_wav(creator_id, tmp_audio_wm.name)
        
        # 3. Dùng FFmpeg ghép audio
        cmd = ["ffmpeg", "-y", "-i", tmp_video_in.name, "-i", tmp_audio_wm.name]
        
        if has_audio:
            # Trộn 2 tiếng, độ dài bằng file video đầu vào
            cmd.extend([
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-cutoff", "20000"
            ])
        else:
            # Không có audio gốc, dùng luôn audio watermark. Cắt độ dài bằng video.
            if video_duration:
                cmd.extend(["-t", str(video_duration)])
            cmd.extend([
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-cutoff", "20000"
            ])
            
        cmd.extend(["-loglevel", "error", tmp_video_out.name])
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace") if hasattr(result.stderr, 'decode') else str(result.stderr)
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
        if os.path.exists(tmp_video_in.name): os.remove(tmp_video_in.name)
        if os.path.exists(tmp_audio_wm.name): os.remove(tmp_audio_wm.name)
        if os.path.exists(tmp_video_out.name): os.remove(tmp_video_out.name)
        
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
    tmp_video_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_audio_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
             
    tmp_video_in.write(video_bytes)
    
    # Đóng file handle trên Windows
    tmp_video_in.close()
    tmp_audio_out.close()
        
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
        
        # HLS và amix có thể làm lệch pha (time shift) dẫn đến cắt sai bit
        # Ta sẽ quét 10 pha (offset) khác nhau để tìm ra pha chuẩn nhất khớp với header
        best_data_bits = None
        
        for phase in range(10):
            offset = int((phase / 10.0) * chunk_size)
            if offset + chunk_size > len(audio_data):
                break
                
            audio_shifted = audio_data[offset:]
            num_chunks = len(audio_shifted) // chunk_size
            powers = []
            
            for i in range(num_chunks):
                chunk = audio_shifted[i*chunk_size : (i+1)*chunk_size]
                # Thực hiện Fast Fourier Transform
                fft_result = np.fft.rfft(chunk)
                freqs = np.fft.rfftfreq(chunk_size, 1.0/sample_rate)
                
                # Tìm index của tần số gần 18000Hz nhất
                idx_18k = np.argmin(np.abs(freqs - freq_target))
                
                # Tính năng lượng quanh dải 18kHz
                power = np.sum(np.abs(fft_result[max(0, idx_18k-2) : min(len(fft_result), idx_18k+3)])**2)
                powers.append(power)
                
            if not powers:
                continue
                
            # 4. Xác định ngưỡng (threshold) để phân biệt bit 1 và bit 0
            sorted_powers = np.sort(powers)
            top_10_percent = sorted_powers[int(len(sorted_powers)*0.9):]
            if len(top_10_percent) == 0:
                threshold = np.mean(powers) * 1.5
            else:
                threshold = np.median(top_10_percent) * 0.3 # 30% của peak
            
            binary_sequence = "".join(["1" if p > threshold else "0" for p in powers])
            
            # 5. Tìm chuỗi header '10101010' để đồng bộ (sync)
            header = "10101010"
            header_idx = binary_sequence.find(header)
            
            if header_idx != -1:
                # Tìm thấy header! Offset này là chính xác!
                best_data_bits = binary_sequence[header_idx + len(header):]
                break
        
        if best_data_bits is None:
            raise ValueError("Không tìm thấy tín hiệu watermark trong video (Không thấy header)")
            
        data_bits = best_data_bits
        
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
        if os.path.exists(tmp_video_in.name): os.remove(tmp_video_in.name)
        if os.path.exists(tmp_audio_out.name): os.remove(tmp_audio_out.name)
