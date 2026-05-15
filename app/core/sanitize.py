"""
Input Sanitizer — Làm sạch dữ liệu đầu vào, chống XSS.

Gọi trước khi xử lý bất kỳ text nào từ user.
"""

import bleach


def clean_text(text: str) -> str:
    """
    Loại bỏ HTML tags khỏi text input.

    "<script>alert('xss')</script>Hello" → "alert('xss')Hello"
    "<b>Bold</b> text" → "Bold text"
    "Normal text" → "Normal text" (không thay đổi)
    """
    return bleach.clean(text, tags=[], strip=True).strip()
