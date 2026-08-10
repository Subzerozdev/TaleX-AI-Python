import tempfile
import cv2
import numpy as np
from blind_watermark import WaterMark

WM_SHAPE_LENGTH = 64

def string_to_bits(s: str) -> list:
    s = s.encode('ascii', 'ignore').decode('ascii')
    payload = f"ID: {s} - Website: talex.pro.vn"
    payload = payload.ljust(WM_SHAPE_LENGTH, ' ')[:WM_SHAPE_LENGTH]
    # Convert character by character to 8 bits
    return [int(b) for b in ''.join([f"{ord(c):08b}" for c in payload])]

def bits_to_string(bits: list) -> str:
    chars = []
    # Process every 8 bits as a character
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        # Convert True/False or 1.0/0.0 to '1' or '0'
        bit_str = ''.join(['1' if float(b) > 0.5 else '0' for b in byte])
        char_code = int(bit_str, 2)
        chars.append(chr(char_code))
    return ''.join(chars).strip()

def _ensure_3_channels(image_bytes: bytes) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    success, encoded_image = cv2.imencode('.png', img)
    return encoded_image.tobytes()

def test():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite('test_orig.png', img)
    
    with open('test_orig.png', 'rb') as f:
        orig_bytes = f.read()

    processed_bytes = _ensure_3_channels(orig_bytes)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_in.write(processed_bytes)
        tmp_in.flush()
        
        bwm = WaterMark(password_img=1, password_wm=1)
        bwm.read_img(tmp_in.name)
        
        bits = string_to_bits('9iP')
        bwm.read_wm(bits, mode='bit')
        
        bwm.embed(tmp_out.name)
        
        with open(tmp_out.name, "rb") as f:
            embedded_bytes = f.read()
            
    # Simulate JPEG compression
    img_bgr = cv2.imdecode(np.frombuffer(embedded_bytes, np.uint8), cv2.IMREAD_COLOR)
    success, jpg_bytes = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    processed_jpg_bytes = _ensure_3_channels(jpg_bytes.tobytes())
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ext:
        tmp_ext.write(processed_jpg_bytes)
        tmp_ext.flush()
        
        bwm2 = WaterMark(password_img=1, password_wm=1)
        try:
            extracted_bits = bwm2.extract(tmp_ext.name, wm_shape=WM_SHAPE_LENGTH*8, mode='bit')
            extracted_str = bits_to_string(extracted_bits)
            safe_str = extracted_str.encode('ascii', 'replace').decode('ascii')
            print("Extracted clean from JPG (mode=bit):", safe_str)
        except Exception as e:
            print("Extraction failed:", e)

test()
