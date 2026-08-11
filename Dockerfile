# ===== Stage 1: Build dependencies =====
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# sentence-transformers kéo theo torch — pip mặc định cài bản CÓ CUDA (vài GB, kèm hàng
# loạt gói nvidia-*) dù VPS này không có GPU, từng làm hết sạch dung lượng đĩa lúc build
# (lỗi thật: "no space left on device"). Cài torch bản CPU-only TRƯỚC để khi
# sentence-transformers yêu cầu torch, pip thấy đã thỏa mãn, không tự tải lại bản CUDA.
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

# Cài FFmpeg và Tesseract OCR (cần cho Content ID — trích xuất watermark từ video)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy thư viện đã cài từ stage 1
COPY --from=builder /install /usr/local

# Download embedding model lúc build — đặt TRƯỚC COPY . . để cache layer này chỉ bị
# invalidate khi requirements.txt đổi, không phải mỗi lần sửa code app (COPY . . luôn
# invalidate mọi layer phía sau nó, kể cả layer không phụ thuộc gì vào source code).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code
COPY . .

# Tạo thư mục cho logs và ChromaDB data
RUN mkdir -p logs chroma_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
