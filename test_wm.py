import tempfile
from blind_watermark import WaterMark
import numpy as np
import cv2

WM_SHAPE_LENGTH = 64

def _pad_id(creator_id: str) -> str:
    creator_id_ascii = creator_id.encode('ascii', 'ignore').decode('ascii')
    payload = f"ID: {creator_id_ascii} - Website: talex.pro.vn"
    return payload.ljust(WM_SHAPE_LENGTH, ' ')[:WM_SHAPE_LENGTH]

def _unpad_id(padded_id: str) -> str:
    return padded_id.strip()

def test():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite('test.png', img)

    creator_id = '9iP'
    padded_id = _pad_id(creator_id)

    bwm = WaterMark(password_img=1, password_wm=1)
    bwm.read_img('test.png')
    bwm.read_wm(padded_id, mode='str')
    
    len_wm = len(bwm.wm_bit)
    print("Actual embedded len_wm:", len_wm)
    
    bwm.embed('test_wm.png')

    bwm2 = WaterMark(password_img=1, password_wm=1)
    
    wm_shape_exact = (WM_SHAPE_LENGTH * 8) - 1
    print("Trying to extract with wm_shape:", wm_shape_exact)
    
    try:
        extracted = bwm2.extract('test_wm.png', wm_shape=wm_shape_exact, mode='str')
        print("Extracted raw:", extracted)
        print("Extracted clean:", _unpad_id(extracted))
    except Exception as e:
        print("Error during extraction:", e)

test()
