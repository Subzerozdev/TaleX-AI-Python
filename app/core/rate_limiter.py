"""
Rate Limiter — Giới hạn request theo IP.

Ngăn user spam request → bảo vệ Gemini quota + server.
Giống @RateLimiter trong Spring Boot.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Limiter lấy IP từ request để đếm
limiter = Limiter(key_func=get_remote_address)
