# ===== Stage 1: Build dependencies =====
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

# Cài FFmpeg (cần cho Content ID — trích xuất frames từ video)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy thư viện đã cài từ stage 1
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Download embedding model lúc build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Tạo thư mục cho logs và ChromaDB data
RUN mkdir -p logs chroma_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
