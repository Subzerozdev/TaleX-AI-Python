"""
Cấu hình chung cho app.
Đọc biến môi trường từ file .env (giống application.properties trong Spring Boot).
"""

import os
from dotenv import load_dotenv

# Load file .env vào environment variables
load_dotenv()


class Settings:
    """Tập trung mọi config ở 1 chỗ — dễ quản lý, dễ thay đổi."""

    # Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Server
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

    # Rate Limiting
    RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "10/minute")
    RATE_LIMIT_SEARCH: str = os.getenv("RATE_LIMIT_SEARCH", "30/minute")

    # Timeouts (giây)
    GEMINI_TIMEOUT: int = 10
    CHROMADB_TIMEOUT: int = 5

    # Search
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 50

    # Chat history
    MAX_HISTORY_MESSAGES: int = 20

    # Milvus (Vector DB cho Content ID fingerprint)
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))

    # Fingerprint
    FINGERPRINT_FPS: int = int(os.getenv("FINGERPRINT_FPS", "1"))
    # 0.90 thay vì 0.85 mặc định — tranh vẽ (màu phẳng, nét đơn giản) dễ bị pHash coi
    # là "giống nhau" hơn ảnh chụp thật dù nội dung khác hẳn, cần ngưỡng chặt hơn để
    # giảm false positive (rủi ro chấp nhận được vì giờ đã route qua Staff review,
    # không còn tự động chặn cứng — xem ContentPipelineServiceImpl.handleCopyrightResult).
    FINGERPRINT_SIMILARITY_THRESHOLD: float = float(os.getenv("FINGERPRINT_SIMILARITY_THRESHOLD", "0.90"))
    FINGERPRINT_MIN_MATCH_SECONDS: int = int(os.getenv("FINGERPRINT_MIN_MATCH_SECONDS", "5"))
    FINGERPRINT_MAX_GAP_SECONDS: int = int(os.getenv("FINGERPRINT_MAX_GAP_SECONDS", "2"))
    FINGERPRINT_MAX_FILE_SIZE_MB: int = int(os.getenv("FINGERPRINT_MAX_FILE_SIZE_MB", "100"))
    # Ảnh chỉ có ĐÚNG 1 vector truy vấn (khác video có tới hàng trăm frame), nên top_k thấp
    # (như video dùng =3) giới hạn cả mạng lưới chỉ còn tối đa 3 ứng viên — nếu 1 kẻ đạo nội
    # dung đăng lại cùng ảnh ở 3+ tài khoản khác, nguồn gốc thật có thể bị đẩy khỏi top_k.
    # Ảnh rẻ hơn video nhiều (1 vector vs tới 300) nên mở rộng top_k riêng cho ảnh không tốn kém.
    FINGERPRINT_IMAGE_TOP_K: int = int(os.getenv("FINGERPRINT_IMAGE_TOP_K", "20"))
    # Giới hạn tổng số frame trích xuất để tạo fingerprint — không có cap này, video dài
    # (vd 1 tiếng) ở FPS mặc định sinh ra hàng nghìn frame, mỗi frame là 1 vector Milvus,
    # làm chậm xử lý và phình dữ liệu không cần thiết. Giống REKOGNITION_MAX_FRAMES bên
    # kiểm duyệt nhưng nhiều hơn — fingerprint cần rải đều khắp video để bắt đúng đoạn
    # trùng, không chỉ lấy mẫu để phân loại như kiểm duyệt.
    FINGERPRINT_MAX_FRAMES: int = int(os.getenv("FINGERPRINT_MAX_FRAMES", "300"))

    # Kafka (Aiven)
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    KAFKA_SSL_CAFILE: str = os.getenv("KAFKA_SSL_CAFILE", "")
    KAFKA_SSL_CERTFILE: str = os.getenv("KAFKA_SSL_CERTFILE", "")
    KAFKA_SSL_KEYFILE: str = os.getenv("KAFKA_SSL_KEYFILE", "")
    KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "python-content-pipeline-group")
    # Cho phép local dev dùng topic riêng (vd "-local"), tách biệt hoàn toàn khỏi VPS dùng
    # chung 1 cụm Kafka Aiven — mặc định rỗng nên VPS không cần đổi gì.
    KAFKA_TOPIC_SUFFIX: str = os.getenv("KAFKA_TOPIC_SUFFIX", "")

    # AWS S3 + Rekognition
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-southeast-1")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "talex-media-139139347126")

    # Rekognition Content Moderation
    REKOGNITION_CONFIDENCE_THRESHOLD: float = float(os.getenv("REKOGNITION_CONFIDENCE_THRESHOLD", "80.0"))
    # Ngưỡng riêng, THẤP HƠN, cho nhóm Violence/Visually Disturbing — Rekognition tự phân
    # loại nội dung Animated/Illustrated khác ảnh chụp thật (xem docs AWS), dấu hiệu bạo lực
    # (máu, vết thương, vũ khí) trên tranh vẽ manga/manhwa bị cách điệu hóa nên confidence
    # thường thấp hơn hẳn so với ảnh thật — dùng chung ngưỡng 80% với Nudity/Suggestive
    # (vẫn rõ nét dù là tranh vẽ) khiến nội dung bạo lực lọt qua kiểm duyệt.
    REKOGNITION_VIOLENCE_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("REKOGNITION_VIOLENCE_CONFIDENCE_THRESHOLD", "60.0"))
    REKOGNITION_MAX_FRAMES: int = int(os.getenv("REKOGNITION_MAX_FRAMES", "30"))
    MODERATION_FRAME_INTERVAL: float = float(os.getenv("MODERATION_FRAME_INTERVAL", "2.0"))

    # MongoDB
    MONGO_URI: str = os.getenv("MONGO_URI", "")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "TaleX-FeatureStore")


# Singleton — toàn app dùng chung 1 instance
settings = Settings()
