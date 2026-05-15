"""
Prompt Templates — Quản lý tất cả prompt gửi cho Gemini.

Tập trung prompt ở 1 file → dễ chỉnh sửa, không hardcode rải rác.
Giống việc tách SQL query ra file riêng trong Spring Boot.
"""

# System prompt cho chatbot — luật chơi cho Gemini
CHAT_SYSTEM_PROMPT = """Bạn là trợ lý AI của TaleX — nền tảng video truyện tranh và hoạt hình ngắn.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời câu hỏi liên quan đến: video, truyện tranh, hoạt hình, anime, manga, manhwa, thể loại, creator, series.
2. Nếu câu hỏi KHÔNG liên quan → từ chối lịch sự, gợi ý quay lại chủ đề video.
3. Trả lời bằng tiếng Việt, thân thiện, tự nhiên.
4. Khi gợi ý video, chỉ chọn từ danh sách video được cung cấp — KHÔNG bịa video không có trong danh sách.
5. Giải thích ngắn gọn tại sao gợi ý video đó.

ĐỊNH DẠNG TRẢ VỀ (BẮT BUỘC JSON):
{
  "is_relevant": true hoặc false,
  "reply": "câu trả lời thân thiện bằng tiếng Việt",
  "video_ids": [danh sách video_id phù hợp, rỗng nếu không liên quan]
}"""


def build_chat_prompt(
    message: str,
    videos_context: str,
    preferences: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    """
    Xây dựng prompt đầy đủ cho chatbot.

    Args:
        message: Câu hỏi của user.
        videos_context: Danh sách video tìm được từ ChromaDB (text).
        preferences: Sở thích user (genres, liked_videos...) — None nếu guest.
        history: Lịch sử hội thoại — None nếu câu đầu tiên.
    """
    parts = []

    # Thêm context sở thích nếu có (user đã đăng nhập)
    if preferences:
        genres = preferences.get("favorite_genres", [])
        liked = preferences.get("liked_video_ids", [])
        parts.append(f"SỞ THÍCH USER: Thể loại yêu thích: {genres}. Video đã thích: {liked}.")

    # Thêm lịch sử hội thoại nếu có
    if history:
        history_text = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Bot'}: {msg['content']}"
            for msg in history[-20:]  # giới hạn 20 messages gần nhất
        )
        parts.append(f"LỊCH SỬ HỘI THOẠI:\n{history_text}")

    # Danh sách video tìm được
    parts.append(f"DANH SÁCH VIDEO TRÊN TALEX:\n{videos_context}")

    # Câu hỏi của user
    parts.append(f"CÂU HỎI CỦA USER: {message}")

    # Nhắc lại format trả về
    parts.append("Hãy trả về JSON đúng format đã quy định. CHỈ trả JSON, không thêm text ngoài.")

    return "\n\n".join(parts)


# Prompt cho auto-tagging
CONTENT_ANALYZE_PROMPT = """Phân tích nội dung video sau và trả về JSON:

Title: {title}
Description: {description}

Trả về JSON (CHỈ JSON, không thêm text):
{{
  "suggested_tags": ["tag1", "tag2", ...],
  "mood": "intense/wholesome/dark/comedic/dramatic/mysterious/romantic",
  "age_rating": "7+/13+/16+/18+",
  "content_type": "anime_series/manga_video/manhwa_video/animated_short/amv"
}}"""


# Prompt cho content moderation
MODERATION_PROMPT = """Kiểm tra nội dung video sau có vi phạm quy tắc cộng đồng không:

Title: {title}
Description: {description}
Tags: {tags}

Kiểm tra các vi phạm:
- Bạo lực quá mức (excessive violence)
- Nội dung người lớn (adult content)
- Vi phạm bản quyền (copyright — chứa tên thương hiệu nổi tiếng không được phép)
- Ngôn từ thù ghét (hate speech)
- Spam / clickbait

Trả về JSON (CHỈ JSON, không thêm text):
{{
  "is_safe": true hoặc false,
  "confidence": 0.0 đến 1.0,
  "flags": ["tên vi phạm nếu có"],
  "reason": "giải thích ngắn gọn",
  "suggestion": "gợi ý cho Staff"
}}"""
