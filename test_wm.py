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

def _ensure_3_channels(image_bytes: bytes) -> bytes:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    success, encoded_image = cv2.imencode('.png', img)
    return encoded_image.tobytes()

def test():
    # 1. Create original 3-channel image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite('test_orig.png', img)
    
    with open('test_orig.png', 'rb') as f:
        orig_bytes = f.read()

    # 2. Embed
    processed_bytes = _ensure_3_channels(orig_bytes)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_in.write(processed_bytes)
        tmp_in.flush()
        
        bwm = WaterMark(password_img=1, password_wm=1)
        bwm.read_img(tmp_in.name)
        bwm.read_wm(_pad_id('9iP'), mode='str')
        bwm.embed(tmp_out.name)
        
        with open(tmp_out.name, "rb") as f:
            embedded_bytes = f.read()
            
    # 3. Simulate S3 / CloudFront doing JPEG compression, or user saving as JPG
    # We will read the PNG bytes, and write it out as a JPEG file with 95% quality.
    img_bgr = cv2.imdecode(np.frombuffer(embedded_bytes, np.uint8), cv2.IMREAD_COLOR)
    success, jpg_bytes = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # 4. Simulate user uploading the JPG to Admin
    processed_jpg_bytes = _ensure_3_channels(jpg_bytes.tobytes())
    
    # 5. Extract
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_ext:
        tmp_ext.write(processed_jpg_bytes)
        tmp_ext.flush()
        
        bwm2 = WaterMark(password_img=1, password_wm=1)
        try:
            extracted = bwm2.extract(tmp_ext.name, wm_shape=511, mode='str')
            print("Extracted clean from JPG:", _unpad_id(extracted))
        except Exception as e:
            print("Extraction failed:", e)

test()
