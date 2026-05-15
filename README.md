# TaleX AI Service

AI Service cho nền tảng TaleX — chatbot, search, tagging, moderation.

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **LLM**: Google Gemini 2.5 Flash
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Python**: 3.12+

## Quick Start

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Tạo file .env từ mẫu
cp .env.example .env
# Sửa GEMINI_API_KEY trong .env

# 3. Chạy app
uvicorn app.main:app --reload

# 4. Mở Swagger UI
# http://localhost:8000/docs
```

## Docker

```bash
# Build
docker build -t talex-ai-service .

# Run
docker run -d -p 8000:8000 --env-file .env --name talex-ai talex-ai-service
```

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/search` | Tìm kiếm video (semantic search) |
| POST | `/api/v1/chat` | Chatbot (RAG + LLM) |
| POST | `/api/v1/sync` | Đồng bộ video từ Spring Boot |
| POST | `/api/v1/content/analyze` | Auto-tagging video |
| POST | `/api/v1/moderation/check` | Kiểm duyệt nội dung |
