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

WM_SHAPE_LENGTH_OLD = 64
WM_SHAPE_LENGTH_NEW = 128

def _pad_id_v2(creator_id: str, viewer_id: str = "") -> str:
    """Đệm chuỗi 128 bytes chứa cả 2 ID."""
    creator_id_ascii = creator_id.encode('ascii', 'ignore').decode('ascii')
    viewer_id_ascii = viewer_id.encode('ascii', 'ignore').decode('ascii') if viewer_id else "NONE"
    
    payload = f"C:{creator_id_ascii}|V:{viewer_id_ascii}"
    return payload.ljust(WM_SHAPE_LENGTH_NEW, ' ')[:WM_SHAPE_LENGTH_NEW]

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

def embed_image_watermark(image_bytes: bytes, creator_id: str, viewer_id: str = "") -> bytes:
    """Nhúng watermark ẩn vào hình ảnh (DWT-DCT-SVD)."""
    padded_id = _pad_id_v2(creator_id, viewer_id)
    
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

def extract_image_watermark(image_bytes: bytes) -> dict:
    """Trích xuất watermark ẩn từ hình ảnh. Trả về dict chứa creator_id và viewer_id."""
    processed_image_bytes = _ensure_3_channels(image_bytes)
    
    tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_in.write(processed_image_bytes)
    tmp_in.close()
        
    try:
        bwm = WaterMark(
            password_img=settings.WATERMARK_PASSWORD_IMG, 
            password_wm=settings.WATERMARK_PASSWORD_WM
        )
        
        # 1. Thử quét với format mới (128 bytes) chứa cả 2 ID
        try:
            wm_shape_exact = (WM_SHAPE_LENGTH_NEW * 8) - 1
            extracted = bwm.extract(tmp_in.name, wm_shape=wm_shape_exact, mode='str')
            extracted_clean = extracted.strip()
            
            if extracted_clean.startswith("C:"):
                parts = extracted_clean.split("|V:")
                creator_id = parts[0][2:] if len(parts) > 0 else None
                viewer_id = parts[1] if len(parts) > 1 else None
                if viewer_id == "NONE":
                    viewer_id = None
                return {"creator_id": creator_id, "viewer_id": viewer_id}
        except Exception:
            pass
            
        # 2. Quét dự phòng với format cũ (64 bytes) chỉ chứa Creator ID
        try:
            wm_shape_exact = (WM_SHAPE_LENGTH_OLD * 8) - 1
            extracted = bwm.extract(tmp_in.name, wm_shape=wm_shape_exact, mode='str')
            extracted_clean = extracted.strip()
            
            if extracted_clean.startswith("ID:"):
                c_id = extracted_clean.replace("ID: ", "").split(" - ")[0]
                return {"creator_id": c_id, "viewer_id": None}
        except Exception:
            pass
            
        return {"creator_id": None, "viewer_id": None}
        
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
    freq = 18000.0  # 18kHz
    sample_rate = 44100
    bit_duration = 0.02  # 50 bits per second (nhanh gấp 5 lần so với cũ)
    
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

def _get_ffmpeg_path():
    """Lấy đường dẫn FFmpeg, fallback về thư mục Desktop nếu không có trong PATH."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    desktop_ffmpeg = r"C:\Users\ttinh\Desktop\ffmpeg-2026-06-08-git-6028720d70-full_build\bin\ffmpeg.exe"
    if os.path.exists(desktop_ffmpeg):
        return desktop_ffmpeg
    return "ffmpeg"

def embed_ab_watermark_hls(video_bytes: bytes, output_dir: str):
    """
    Tạo 2 phiên bản HLS A và B từ video gốc.
    Version A: Có đóng dấu (Pattern A)
    Version B: Không đóng dấu (hoặc Pattern B)
    """
    tmp_video_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_in.write(video_bytes)
    tmp_video_in.close()

    dir_a = os.path.join(output_dir, "version_A")
    dir_b = os.path.join(output_dir, "version_B")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    ffmpeg_bin = _get_ffmpeg_path()
    
    try:
        import platform
        font_param = ""
        if platform.system() == "Windows":
            font_path = "C\\\\:/Windows/Fonts/arial.ttf"
            font_param = f"fontfile={font_path}:"

        # Lệnh Version A (Có Pattern A - ví dụ là 1 text nhỏ ở góc phải)
        # Sử dụng font mặc định của FFmpeg hoặc arial trên Windows để tránh lỗi thiếu font
        cmd_a = [
            ffmpeg_bin, "-y", "-i", tmp_video_in.name,
            "-vf", f"drawtext={font_param}text='talex.pro.vn':x=W-tw-10:y=10:fontsize=24:fontcolor=white@0.8",
            "-c:v", "libx264", "-preset", "fast",
            "-force_key_frames", "expr:gte(t,n_forced*4)",
            "-g", "120", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-hls_time", "4", "-hls_playlist_type", "vod",
            "-f", "hls", os.path.join(dir_a, "playlist.m3u8")
        ]
        
        # Lệnh Version B (Không có watermark)
        cmd_b = [
            ffmpeg_bin, "-y", "-i", tmp_video_in.name,
            "-c:v", "libx264", "-preset", "fast",
            "-force_key_frames", "expr:gte(t,n_forced*4)",
            "-g", "120", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-hls_time", "4", "-hls_playlist_type", "vod",
            "-f", "hls", os.path.join(dir_b, "playlist.m3u8")
        ]

        logger.info("Đang render Version A HLS...")
        res_a = subprocess.run(cmd_a, capture_output=True)
        if res_a.returncode != 0:
            raise RuntimeError(f"FFmpeg A failed: {res_a.stderr.decode('utf-8', errors='replace')}")

        logger.info("Đang render Version B HLS...")
        res_b = subprocess.run(cmd_b, capture_output=True)
        if res_b.returncode != 0:
            raise RuntimeError(f"FFmpeg B failed: {res_b.stderr.decode('utf-8', errors='replace')}")
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy A/B HLS: {e}")
        raise e
    finally:
        if os.path.exists(tmp_video_in.name): os.remove(tmp_video_in.name)
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

def _extract_ook_from_audio(audio_data: np.ndarray, sample_rate: int, freq_target: float, bit_duration: float) -> str:
    """Phân tích OOK từ mảng audio theo tần số và độ dài bit."""
    chunk_size = int(sample_rate * bit_duration)
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
            fft_result = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(chunk_size, 1.0/sample_rate)
            
            idx_freq = np.argmin(np.abs(freqs - freq_target))
            power = np.sum(np.abs(fft_result[max(0, idx_freq-2) : min(len(fft_result), idx_freq+3)])**2)
            powers.append(power)
            
        if not powers:
            continue
            
        sorted_powers = np.sort(powers)
        top_10_percent = sorted_powers[int(len(sorted_powers)*0.9):]
        if len(top_10_percent) == 0:
            threshold = np.mean(powers) * 1.5
        else:
            threshold = np.median(top_10_percent) * 0.3
        
        binary_sequence = "".join(["1" if p > threshold else "0" for p in powers])
        
        header = "10101010"
        header_idx = binary_sequence.find(header)
        
        if header_idx != -1:
            best_data_bits = binary_sequence[header_idx + len(header):]
            break
    
    if best_data_bits is None:
        return None
        
    extracted_str = _binary_to_string(best_data_bits)
    clean_id = ''.join(c for c in extracted_str if 32 <= ord(c) <= 126)
    return clean_id

def extract_ab_watermark_hls(video_bytes: bytes) -> dict:
    """
    Trích xuất Viewer ID từ một đoạn video bị quay lén (đã áp dụng A/B Watermarking).
    Thuật toán:
    1. Trích xuất 1 frame mỗi 4 giây bằng FFmpeg (1 lệnh duy nhất cho lẹ).
    2. Dùng OCR (PyTesseract) quét tối đa 32 frames đầu tiên (để tránh timeout).
    3. Gom kết quả quét thành chuỗi nhị phân (Ví dụ: 10101).
    4. Giải mã chuỗi nhị phân thành ViewerID gốc.
    """
    import pytesseract
    import platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    from PIL import Image
    import cv2
    import math
    import glob

    tmp_video_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_in.write(video_bytes)
    tmp_video_in.close()
    
    tmp_dir = tempfile.mkdtemp()
    ffmpeg_bin = _get_ffmpeg_path()
    
    binary_str = ""
    try:
        # 1. Trích xuất frames: 1 frame mỗi 4 giây (fps=1/4)
        # Bắt đầu lấy từ giây thứ 2 (để né cảnh chuyển mờ đầu chunk)
        # Nhưng fps=1/4 sẽ tự động chia đều.
        cmd = [
            ffmpeg_bin, "-y", "-i", tmp_video_in.name,
            "-vf", "fps=1/4", "-q:v", "2",
            os.path.join(tmp_dir, "frame_%04d.jpg")
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Lấy danh sách các frame đã xuất (sắp xếp theo thời gian)
        frames = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.jpg")))
        
        # Chỉ quét tối đa 32 frames (tương đương 32 bits = 2 phút) để tránh timeout API
        frames_to_scan = frames[:32]
        
        if not frames_to_scan:
            return {"creator_id": None, "viewer_id": None}
            
        for frame_path in frames_to_scan:
            img = cv2.imread(frame_path)
            if img is None:
                binary_str += "0"
                continue
                
            # Cắt góc trên bên phải (nơi chứa Pattern A - x=W-tw-10, y=10)
            h, w = img.shape[:2]
            # Pattern nằm ở góc trên bên phải, ta cắt 15% chiều cao và 30% chiều rộng từ mép phải
            roi = img[0:int(h*0.15), int(w*0.7):w] 
            
            # Phóng to ảnh 3 lần để Tesseract dễ đọc chữ nhỏ (fontsize=12)
            roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            
            # Tiền xử lý để tăng khả năng nhận diện chữ trắng
            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Lọc nhiễu và làm nét
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # Threshold động hoặc thấp (vì opacity=0.3 làm chữ có màu xám tối)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Quét OCR với custom config
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config).lower()
            if "talex" in text or "pro" in text or "vn" in text:
                binary_str += "1"
            else:
                binary_str += "0"
                
        # Giả lập: Lấy ra Viewer ID từ chuỗi bit thu được
        viewer_id = None
        if "1" in binary_str:
            viewer_id = f"User_Binary_{binary_str}"
            
        # Lấy thêm Creator ID từ Audio Watermark (sóng siêu âm)
        creator_id = None
        try:
            audio_extracted = extract_video_audio_watermark(video_bytes)
            creator_id = audio_extracted.get("creator_id")
        except Exception as e:
            logger.warning(f"Không thể trích xuất audio watermark: {e}")
            
        return {"creator_id": creator_id, "viewer_id": viewer_id}        
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất A/B HLS: {e}")
        return {"creator_id": None, "viewer_id": None}
    finally:
        if os.path.exists(tmp_video_in.name): 
            os.remove(tmp_video_in.name)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

def extract_video_audio_watermark(video_bytes: bytes) -> dict:
    """
    Trích xuất ID từ âm thanh siêu âm của Video bằng FFT.
    Trả về dict chứa creator_id (18kHz) và viewer_id (20kHz).
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
            
        # 3. Phân tích OOK ở 18kHz (Creator, 0.02s) và 20kHz (Viewer, 0.05s)
        creator_id = _extract_ook_from_audio(audio_data, sample_rate, 18000.0, 0.02)
        viewer_id = _extract_ook_from_audio(audio_data, sample_rate, 20000.0, 0.05)
        
        if not creator_id and not viewer_id:
            raise ValueError("Không tìm thấy tín hiệu watermark nào trong video (Không thấy header)")
            
        return {
            "creator_id": creator_id,
            "viewer_id": viewer_id
        }
        
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
