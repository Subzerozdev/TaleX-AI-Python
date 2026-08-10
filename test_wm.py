import tempfile
from blind_watermark import WaterMark
import numpy as np

def string_to_bits(s):
    # Convert string to exactly len(s)*8 bits
    return [int(b) for b in ''.join([f"{c:08b}" for c in s.encode('utf-8')])]

def bits_to_string(bits):
    # Convert bits back to string
    chars = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b+8]
        chars.append(chr(int(''.join([str(int(bit)) for bit in byte]), 2)))
    return ''.join(chars)

def test():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite('test.png', img)

    wm_str = 'ID: 9iP - Website: talex.pro.vn'
    WM_SHAPE_LENGTH = 64
    padded_id = wm_str.ljust(WM_SHAPE_LENGTH)[:WM_SHAPE_LENGTH]

    # Convert to EXACTLY 512 bits
    bits = string_to_bits(padded_id)
    print("Bit array length:", len(bits))

    bwm = WaterMark(password_img=1, password_wm=1)
    bwm.read_img('test.png')
    bwm.read_wm(bits, mode='bit')
    bwm.embed('test_wm.png')

    bwm2 = WaterMark(password_img=1, password_wm=1)
    extracted_bits = bwm2.extract('test_wm.png', wm_shape=512, mode='bit')
    
    extracted_str = bits_to_string(extracted_bits)
    print("Extracted string:", repr(extracted_str.strip()))

test()
