

from slowapi import Limiter
from slowapi.util import get_remote_address

# Limiter lấy IP từ request để đếm
limiter = Limiter(key_func=get_remote_address)
