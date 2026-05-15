"""
Gemini Client — Gọi Google Gemini API.

Tầng Infrastructure: chỉ lo giao tiếp với Gemini, không chứa business logic.
Giống RestTemplate/WebClient gọi API bên thứ 3 trong Spring Boot.

Áp dụng Dependency Inversion:
  Service gọi gemini_client (abstraction) → không gọi thẳng Gemini SDK.
  Muốn đổi sang OpenAI/DeepSeek → chỉ sửa file này, không sửa service.
"""

import json

from google import genai
from google.genai import types
from loguru import logger

from app.core.config import settings

# Gemini client (singleton)
_client: genai.Client | None = None


def init_gemini() -> None:
    """Khởi tạo Gemini client. Gọi 1 lần khi app khởi động."""
    global _client

    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. LLM features will use fallback.")
        return

    _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    logger.info(f"Gemini client initialized. Model: {settings.GEMINI_MODEL}")


def is_configured() -> bool:
    """Kiểm tra Gemini đã được cấu hình chưa — dùng cho health check."""
    return _client is not None


def generate(prompt: str, system_prompt: str | None = None) -> str:
    """
    Gửi prompt đến Gemini, nhận response text.

    Args:
        prompt: Nội dung chính cần hỏi.
        system_prompt: Luật chơi cho Gemini (system instruction).

    Returns:
        Response text từ Gemini.

    Raises:
        RuntimeError: Khi Gemini chưa init hoặc API lỗi.
    """
    if _client is None:
        raise RuntimeError("Gemini client chưa được khởi tạo.")

    logger.debug(f"Gemini request: model={settings.GEMINI_MODEL}, prompt_len={len(prompt)}")

    config = types.GenerateContentConfig(
        temperature=0.7,       # sáng tạo vừa phải (0.0 = cứng nhắc, 1.0 = rất sáng tạo)
        max_output_tokens=2048,
    )

    if system_prompt:
        config.system_instruction = system_prompt

    response = _client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=config,
    )

    result = response.text
    logger.debug(f"Gemini response: len={len(result)}")
    return result


def generate_json(prompt: str, system_prompt: str | None = None) -> dict:
    """
    Gọi Gemini và parse response thành JSON dict.

    Dùng cho các service cần response có cấu trúc (chat, tagging, moderation).
    Nếu Gemini trả text không phải JSON → raise lỗi để service xử lý fallback.
    """
    raw = generate(prompt, system_prompt)

    cleaned = _extract_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini trả response không phải JSON: {raw[:200]}")
        raise ValueError(f"Không thể parse JSON từ Gemini: {e}") from e


def _extract_json(raw: str) -> str:
    """
    Làm sạch response từ Gemini để parse JSON.

    Gemini có thể trả:
      - JSON thuần: {"key": "value"}
      - Bọc trong code block: ```json\n{...}\n```
      - Có trailing comma: {"ids": [1, 2,]}
    """
    cleaned = raw.strip()

    # Bỏ code block ```json ... ```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()

    # Bỏ trailing commas trước } hoặc ] (lỗi thường gặp của LLM)
    import re
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    return cleaned
