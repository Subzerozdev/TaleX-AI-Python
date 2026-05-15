"""
Retriever — Tìm video liên quan từ ChromaDB.

Đây là bước R (Retrieval) trong RAG pipeline:
  1. Nhận câu hỏi text
  2. Chuyển thành vector (embedding)
  3. Tìm video gần nhất trong ChromaDB
  4. Trả về danh sách video_ids + scores
"""

from app.rag.embeddings import embed_text
from app.rag.vector_store import search_similar


def retrieve_videos(query: str, top_k: int = 10) -> dict:
    """
    Tìm video liên quan với câu hỏi.

    Args:
        query: Câu hỏi của user (ví dụ: "truyện dark fantasy")
        top_k: Số kết quả muốn lấy

    Returns:
        {
            "video_ids": [1, 3, 50],
            "scores": [0.95, 0.88, 0.75],
            "documents": ["text gốc video 1", ...]
        }

    Score: 0.0 = hoàn toàn giống, 1.0 = hoàn toàn khác
           (ChromaDB trả distance, không phải similarity)
    """
    # Bước 1: Chuyển câu hỏi thành vector
    query_vector = embed_text(query)

    # Bước 2: Tìm trong ChromaDB
    results = search_similar(query_vector, top_k=top_k)

    # Bước 3: Chuyển string IDs thành int + tính score
    video_ids = []
    scores = []
    for str_id, distance in zip(results["ids"], results["distances"]):
        video_ids.append(int(str_id))
        # Chuyển distance → similarity score (1 - distance)
        # distance 0.0 → score 1.0 (giống nhất)
        # distance 1.0 → score 0.0 (khác nhất)
        scores.append(round(1.0 - distance, 4))

    return {
        "video_ids": video_ids,
        "scores": scores,
        "documents": results["documents"],
    }
