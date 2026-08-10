import tempfile
from blind_watermark import WaterMark
import numpy as np

def test():
    # create a dummy image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite('test.png', img)

    wm_str = '9iP'
    # pad
    WM_SHAPE_LENGTH = 64
    padded_id = wm_str.ljust(WM_SHAPE_LENGTH)[:WM_SHAPE_LENGTH]

    bwm = WaterMark(password_img=1, password_wm=1)
    bwm.read_img('test.png')
    bwm.read_wm(padded_id, mode='str')
    bwm.embed('test_wm.png')

    bwm2 = WaterMark(password_img=1, password_wm=1)
    # The actual length in bits for a 64 char string is 64*8? 
    # Let's test what bwm.wm_bit is
    print("wm_bit length during embed:", len(bwm.wm_bit))

    extracted = bwm2.extract('test_wm.png', wm_shape=WM_SHAPE_LENGTH, mode='str')
    print("extracted (wm_shape=64):", repr(extracted))

    extracted2 = bwm2.extract('test_wm.png', wm_shape=len(bwm.wm_bit), mode='str')
    print("extracted (wm_shape=actual_bits):", repr(extracted2))

test()
