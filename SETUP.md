# Hướng dẫn chạy và test TaleX AI Service

---

## 1. Yêu cầu

- **Docker Desktop** đã cài và đang chạy
- **Git** (để clone repo)
- Không cần cài Python, FFmpeg, hay bất kỳ thư viện nào — Docker lo hết

---

## 2. Chuẩn bị

### 2.1. Clone repo

```bash
git clone git@github.com:Subzerozdev/TaleX-AI-Python.git
cd TaleX-AI-Python
```

### 2.2. Tạo file .env

```bash
cp .env.example .env
```

Mở file `.env`, sửa `GEMINI_API_KEY`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Lấy API key tại: https://aistudio.google.com → Get API key → Create API key

---

## 3. Chạy ứng dụng

### 3.1. Khởi động (lần đầu)

```bash
docker-compose up --build
```

Lần đầu sẽ mất 5-10 phút vì:
- Download Python image
- Cài thư viện (sentence-transformers, pytorch...)
- Download embedding model (~80MB)
- Download Milvus image

### 3.2. Khởi động (các lần sau)

```bash
docker-compose up
```

Nhanh hơn vì Docker đã cache image.

### 3.3. Chạy ngầm (background)

```bash
docker-compose up -d
```

### 3.4. Xem logs

```bash
# Tất cả containers
docker-compose logs -f

# Chỉ Python app
docker-compose logs -f app

# Chỉ Milvus
docker-compose logs -f milvus
```

### 3.5. Dừng ứng dụng

```bash
docker-compose down
```

### 3.6. Dừng và xóa data (reset hoàn toàn)

```bash
docker-compose down -v
```

---

## 4. Kiểm tra ứng dụng đã chạy

### 4.1. Health check

```bash
curl http://localhost:8000/health
```

Kết quả mong đợi:

```json
{
  "status": "healthy",
  "components": {
    "embedding_model": "loaded",
    "chromadb": "connected",
    "gemini": "configured",
    "milvus": "connected",
    "ffmpeg": "available"
  },
  "video_count": 20,
  "fingerprint_count": 0
}
```

Tất cả components phải là `loaded` / `connected` / `configured` / `available`.

### 4.2. Swagger UI

Mở trình duyệt: **http://localhost:8000/docs**

Swagger UI hiển thị tất cả endpoints, cho phép test trực tiếp trên trình duyệt.

---

## 5. Test các endpoints

### 5.1. Search — Tìm video theo text

**Endpoint:** `POST /api/v1/search`

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Doraemon", "top_k": 5}'
```

Kết quả mong đợi:

```json
{
  "video_ids": [6, 1, 12, 4, 18],
  "scores": [0.4869, 0.259, 0.2294, 0.2159, 0.194]
}
```

Video id=6 (Doraemon) đứng đầu với score cao nhất.

**Test thêm:**

```bash
# Tìm truyện dark fantasy
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "truyen dark fantasy nhan vat manh", "top_k": 5}'

# Tìm tình cảm học đường
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "tinh cam hoc duong", "top_k": 5}'
```

---

### 5.2. Chat — Chatbot AI

**Endpoint:** `POST /api/v1/chat`

**Test 1: Chat đơn giản**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Goi y truyen action hay di"}'
```

Kết quả: reply thân thiện bằng tiếng Việt + video_ids phù hợp.

**Test 2: Chat off-topic (câu hỏi không liên quan)**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Cach nau pho bo nhu the nao"}'
```

Kết quả: `is_relevant: false`, reply từ chối lịch sự, `video_ids: []`.

**Test 3: Chat với preferences (user đã đăng nhập)**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tim truyen hay cho toi",
    "preferences": {
      "favorite_genres": ["romance", "school"],
      "liked_video_ids": [2, 9]
    }
  }'
```

Kết quả: gợi ý cá nhân hóa dựa trên sở thích romance + school.

**Test 4: Chat với history (nhớ context)**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bo thu 2 co bao nhieu tap?",
    "history": [
      {"role": "user", "content": "Goi y truyen action"},
      {"role": "assistant", "content": "1. Solo Leveling 2. Demon Slayer 3. Jujutsu Kaisen"}
    ]
  }'
```

Kết quả: AI hiểu "bộ thứ 2" = Demon Slayer từ context.

---

### 5.3. Sync — Đồng bộ video

**Endpoint:** `POST /api/v1/sync`

**Test 1: Thêm video mới**

```bash
curl -X POST http://localhost:8000/api/v1/sync \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": 99,
    "title": "Dragon Ball Z - Goku vs Frieza",
    "description": "Goku Super Saiyan chien dau voi Frieza",
    "tags": ["action", "battle", "classic"],
    "action": "create"
  }'
```

**Test 2: Tìm video vừa thêm**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Dragon Ball", "top_k": 3}'
```

Video 99 phải đứng đầu kết quả.

**Test 3: Xóa video**

```bash
curl -X POST http://localhost:8000/api/v1/sync \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": 99,
    "title": "x",
    "description": "x",
    "tags": ["x"],
    "action": "delete"
  }'
```

---

### 5.4. Content Analyze — Auto-tagging

**Endpoint:** `POST /api/v1/content/analyze`

```bash
curl -X POST http://localhost:8000/api/v1/content/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Dark Knight Ep 5 - Cuoc chien cuoi cung",
    "description": "Hiep si bong dem doi dau ke thu manh nhat trong tran chien quyet dinh."
  }'
```

Kết quả mong đợi:

```json
{
  "suggested_tags": ["action", "dark", "superhero", ...],
  "mood": "intense",
  "age_rating": "16+",
  "content_type": "anime_series"
}
```

---

### 5.5. Moderation — Kiểm duyệt nội dung

**Endpoint:** `POST /api/v1/moderation/check`

**Test 1: Nội dung an toàn**

```bash
curl -X POST http://localhost:8000/api/v1/moderation/check \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Doraemon - Bao boi than ky",
    "description": "Doraemon giup Nobita giai quyet van de bang bao boi than ky.",
    "tags": ["comedy", "kids", "adventure"]
  }'
```

Kết quả: `is_safe: true`.

**Test 2: Nội dung vi phạm bản quyền**

```bash
curl -X POST http://localhost:8000/api/v1/moderation/check \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Marvel Spider-Man Full Movie Leaked",
    "description": "Watch the full leaked movie of Marvel Spider-Man. Download free now!",
    "tags": ["marvel", "leaked", "free download"]
  }'
```

Kết quả: `is_safe: false`, flags chứa copyright + spam.

---

### 5.6. Fingerprint — Content ID (kiểm tra bản quyền video/ảnh)

**Endpoint:** `POST /api/v1/fingerprint/process`

Test bằng Swagger UI dễ hơn curl vì cần upload file.

**Cách test trên Swagger UI:**

1. Mở http://localhost:8000/docs
2. Tìm endpoint `POST /api/v1/fingerprint/process`
3. Bấm "Try it out"
4. Nhập `media_id`: 1
5. Chọn file video (mp4) hoặc ảnh (png/jpg)
6. Bấm "Execute"

**Test bằng curl:**

```bash
# Upload video gốc
curl -X POST http://localhost:8000/api/v1/fingerprint/process \
  -F "media_id=1" \
  -F "file=@test_video_original.mp4"

# Upload video copy (sẽ phát hiện trùng)
curl -X POST http://localhost:8000/api/v1/fingerprint/process \
  -F "media_id=2" \
  -F "file=@test_video_copy.mp4"

# Upload ảnh
curl -X POST http://localhost:8000/api/v1/fingerprint/process \
  -F "media_id=3" \
  -F "file=@test_image.png"
```

**Xem fingerprint đã lưu:**

```bash
curl http://localhost:8000/api/v1/fingerprint/1
```

**Xóa fingerprint:**

```bash
curl -X DELETE http://localhost:8000/api/v1/fingerprint/1
```

---

### 5.7. Validation — Test unhappy cases

**Message quá ngắn:**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "a"}'
```

Kết quả: `422 — String should have at least 2 characters`.

**Sync action không hợp lệ:**

```bash
curl -X POST http://localhost:8000/api/v1/sync \
  -H "Content-Type: application/json" \
  -d '{"video_id": 1, "title": "x", "description": "x", "tags": ["x"], "action": "invalid"}'
```

Kết quả: `422 — Input should be 'create', 'update' or 'delete'`.

**XSS trong search:**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "<script>alert(1)</script>Doraemon"}'
```

Kết quả: HTML bị strip, tìm "Doraemon" bình thường.

---

## 6. Rebuild khi sửa code

Vì chạy Docker, mỗi lần sửa code Python cần rebuild:

```bash
# Rebuild và restart
docker-compose up --build

# Hoặc rebuild chỉ app (không rebuild Milvus)
docker-compose build app && docker-compose up
```

---

## 7. Troubleshooting

### Milvus không kết nối

```bash
# Kiểm tra Milvus container
docker-compose logs milvus

# Restart Milvus
docker-compose restart milvus
```

### Port 8000 đã bị chiếm

```bash
# Tìm process dùng port 8000
netstat -ano | findstr 8000

# Kill process
taskkill /PID <pid> /F
```

### Muốn reset toàn bộ data

```bash
docker-compose down -v
docker-compose up --build
```

### Gemini trả lỗi 503

API key hết quota hoặc Gemini quá tải. App vẫn chạy bình thường với fallback responses. Tạo key mới tại https://aistudio.google.com nếu cần.
