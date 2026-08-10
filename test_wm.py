import tempfile
import cv2
import numpy as np
from blind_watermark import WaterMark

WM_SHAPE_LENGTH = 64

def _pad_id(creator_id: str) -> str:
    creator_id_ascii = creator_id.encode('ascii', 'ignore').decode('ascii')
    payload = f"ID: {creator_id_ascii} - Website: talex.pro.vn"
    return payload.ljust(WM_SHAPE_LENGTH, ' ')[:WM_SHAPE_LENGTH]

def _unpad_id(padded_id: str) -> str:
    return padded_id.strip()

def _ensure_3_channels_jpg(image_bytes: bytes) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # Drop alpha for JPEG
        img = img[:, :, :3]
    success, encoded_image = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return encoded_image.tobytes()

def test():
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite('test_orig.jpg', img)
    
    with open('test_orig.jpg', 'rb') as f:
        orig_bytes = f.read()

    processed_bytes = _ensure_3_channels_jpg(orig_bytes)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        tmp_in.write(processed_bytes)
        tmp_in.flush()
        
        bwm = WaterMark(password_img=1, password_wm=1)
        # Tweak robustness
        bwm.bwm_core.d1 = 50
        bwm.bwm_core.d2 = 30
        
        bwm.read_img(tmp_in.name)
        bwm.read_wm(_pad_id('9iP'), mode='str')
        bwm.embed(tmp_out.name)
        
        # blind_watermark saves it as PNG or whatever extension we give. Wait, cv2.imwrite guesses by extension.
        # It saved as JPG.
        with open(tmp_out.name, "rb") as f:
            embedded_bytes = f.read()
            
    processed_ext_bytes = _ensure_3_channels_jpg(embedded_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_ext:
        tmp_ext.write(processed_ext_bytes)
        tmp_ext.flush()
        
        bwm2 = WaterMark(password_img=1, password_wm=1)
        bwm2.bwm_core.d1 = 50
        bwm2.bwm_core.d2 = 30
        try:
            extracted = bwm2.extract(tmp_ext.name, wm_shape=511, mode='str')
            print("Extracted clean from JPG:", _unpad_id(extracted))
        except Exception as e:
            print("Extraction failed:", e)

test()
