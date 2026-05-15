"""
Health Check endpoint.
Spring Boot gọi định kỳ để kiểm tra AI Service còn sống.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.llm import gemini_client
from app.rag import embeddings, vector_store

router = APIRouter()


@router.get("/health")
def health_check():
    """Trả về trạng thái thật của từng component."""
    return {
        "status": "healthy",
        "components": {
            "embedding_model": "loaded" if embeddings.is_loaded() else "not_loaded",
            "chromadb": "connected" if vector_store.is_connected() else "not_connected",
            "gemini": "configured" if gemini_client.is_configured() else "not_configured",
        },
        "video_count": vector_store.get_video_count(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
