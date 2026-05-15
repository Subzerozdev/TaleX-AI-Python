# ===== Stage 1: Build dependencies =====
# Dùng python slim (nhẹ hơn python full ~500MB)
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy requirements trước → Docker cache layer
# Nếu requirements không đổi → không cần cài lại (nhanh hơn khi rebuild)
COPY requirements.txt .

# Cài thư viện vào folder riêng để copy sang stage 2
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

# Copy thư viện đã cài từ stage 1
COPY --from=builder /install /usr/local

# Copy source code
COPY . .

# Download embedding model lúc build (không phải download lúc chạy container)
# → Container khởi động nhanh hơn, không cần internet lúc chạy
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Tạo thư mục cho logs và ChromaDB data
RUN mkdir -p logs chroma_data

# Port mà FastAPI lắng nghe
EXPOSE 8000

# Lệnh chạy app
# --host 0.0.0.0: cho phép truy cập từ bên ngoài container
# --workers 1: 1 worker đủ cho đồ án (tăng lên nếu cần scale)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
