"""Service to handle recommendations based on series metadata."""

from loguru import logger

from app.rag.embeddings import embed_text
from app.rag.milvus_recommendation_store import insert_series_vector, search_similar_series

def process_series_upsert(series_id: str, title: str, description: str, categories: list[str], tags: list[str], age_rating: str = "", language: str = "") -> list[str]:
    """
    Called when a Series is created/updated (via Debezium CDC).
    Embeds the metadata, stores it in Milvus, and returns similar Series IDs.
    """
    logger.info(f"Processing recommendation embedding for series_id={series_id}")

    # Build document
    cat_str = ", ".join(categories) if categories else ""
    tag_str = ", ".join(tags) if tags else ""
    desc_str = description if description else ""
    title_str = title if title else ""
    age_str = age_rating if age_rating else "Unrated"
    lang_str = language if language else "Unknown"

    document = f"Title: {title_str}. Description: {desc_str}. Categories: {cat_str}. Tags: {tag_str}. Age Rating: {age_str}. Language: {lang_str}."

    try:
        # Embed text
        vector = embed_text(document)

        # Store in Milvus
        insert_series_vector(series_id, vector)

        # Find similarities to recommend
        # We query top 11 to get 10 others (since it will match itself)
        similar_ids = search_similar_series(vector, top_k=11)
        
        # Remove self from recommendations
        if series_id in similar_ids:
            similar_ids.remove(series_id)
        
        # Cap at 10 just in case
        similar_ids = similar_ids[:10]

        logger.info(f"Generated {len(similar_ids)} recommendations for series_id={series_id}")
        return similar_ids

    except Exception as e:
        logger.error(f"Failed to process series recommendation for {series_id}: {e}")
        return []
