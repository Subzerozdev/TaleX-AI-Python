# TaleX AI Service

Python microservice cung cấp các tính năng AI và xử lý media cho nền tảng TaleX — nền tảng video truyện tranh và hoạt hình ngắn.

---

## Kiến trúc

```
┌────────────────────────────────────────────────────────────────┐
│                    TaleX AI Service                            │
│                                                                │
│  ┌─ AI Services ───────────────────────────────────────────┐   │
│  │  Chatbot (RAG)    → ChromaDB + Gemini                   │   │
│  │  Smart Search     → ChromaDB + Embedding                │   │
│  │  Auto-Tagging     → Gemini                              │   │
│  │  Moderation       → Gemini                              │   │
│  │  Sync Data        → Embedding + ChromaDB                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─ Content ID Service ────────────────────────────────────┐   │
│  │  Fingerprint      → FFmpeg + imagehash + Milvus         │   │
│  │  Copyright Check  → Milvus similarity search            │   │
│  │  Segment Matching → Sliding window + merge              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Layered Architecture (3 tầng)

```
Tầng 1 — Routers       Nhận HTTP request, validate input, trả response
Tầng 2 — Services      Business logic, điều phối pipeline
Tầng 3 — Infrastructure
          ├── rag/          ChromaDB + Embedding (text search)
          ├── llm/          Google Gemini (chatbot, tagging, moderation)
          └── fingerprint/  FFmpeg + imagehash + Milvus (Content ID)
```

---

## Tech Stack

### Python App

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Framework | FastAPI + Uvicorn | REST API server |
| LLM | Google Gemini 2.5 Flash | Chatbot reply, auto-tagging, moderation |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) | Text → vector cho semantic search |
| Text Vector DB | ChromaDB | Tìm video theo text (nhúng trong app) |
| Video Processing | FFmpeg | Trích xuất frames + audio từ video |
| Image Hashing | imagehash (pHash) | Tạo perceptual hash cho ảnh/frames |
| Validation | Pydantic v2 | Request/response DTO, tự validate |
| Rate Limiting | SlowAPI | Giới hạn request theo IP |
| Logging | Loguru | Console + file log, auto-rotate |
| Sanitize | Bleach | Chống XSS, strip HTML tags |

### Infrastructure (Docker containers riêng)

| Service | Image | Port | Vai trò |
|---|---|---|---|
| Milvus | milvusdb/milvus:v2.5.10 | 19530 | Vector DB cho video fingerprint (Content ID) |

---

## API Endpoints

### AI Services

| Method | Endpoint | Rate Limit | Mô tả |
|---|---|---|---|
| POST | `/api/v1/search` | 30/min | Tìm video theo ngữ nghĩa (không LLM, nhanh ~200ms) |
| POST | `/api/v1/chat` | 10/min | Chatbot RAG (ChromaDB + Gemini) |
| POST | `/api/v1/sync` | — | Đồng bộ metadata video từ Spring Boot |
| POST | `/api/v1/content/analyze` | 20/min | Auto-tagging video (Gemini) |
| POST | `/api/v1/moderation/check` | 20/min | Kiểm duyệt nội dung text (Gemini) |

### Content ID Service

| Method | Endpoint | Rate Limit | Mô tả |
|---|---|---|---|
| POST | `/api/v1/fingerprint/process` | 5/min | Upload file → fingerprint → kiểm tra bản quyền |
| GET | `/api/v1/fingerprint/{media_id}` | — | Xem thông tin fingerprint đã lưu |
| DELETE | `/api/v1/fingerprint/{media_id}` | — | Xóa fingerprint khi Creator xóa video |

### System

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Health check tất cả components |

---

## Cấu trúc thư mục

```
TaleX-AI-Python/
├── app/
│   ├── main.py                     ← Điểm khởi chạy
│   ├── core/                       ← Config, error handler, rate limiter, logging
│   ├── routers/                    ← Tầng 1: HTTP endpoints
│   ├── services/                   ← Tầng 2: Business logic
│   ├── schemas/                    ← DTO (Pydantic models)
│   ├── rag/                        ← Tầng 3: ChromaDB + Embedding
│   ├── llm/                        ← Tầng 3: Google Gemini
│   └── fingerprint/                ← Tầng 3: FFmpeg + imagehash + Milvus
├── data/
│   └── seed_videos.json            ← 20 video mẫu cho demo
├── docker-compose.yml              ← Python app + Milvus
├── Dockerfile
├── requirements.txt
├── .env.example
└── SETUP.md                        ← Hướng dẫn chạy + test
```

---

## Luồng xử lý chính

### Chatbot (RAG Pipeline)

```
User message → ChromaDB tìm 20 video → Gemini lọc + viết reply → trả video_ids + reply
```

### Content ID (Fingerprint Pipeline)

```
Upload video → FFmpeg cắt frames → pHash tạo vectors → Milvus so sánh → trả violations
```
