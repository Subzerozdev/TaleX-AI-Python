from datetime import datetime
from pathlib import Path
from typing import List
from loguru import logger

import lightgbm as lgb
import pandas as pd

from app.services.recommendation_trainer import RecommendationTrainer
from app.rag.embeddings import embed_text
from app.rag.milvus_recommendation_store import insert_series_vector, search_similar_series

class RecommendationService:
    def __init__(
        self,
        db,
        model_save_path: str = "data/lgb_ranking_model.txt",
        train_data_csv_path: str = "data/train_data.csv",
        train_data_excel_path: str = "data/train_data.xlsx"
    ):
        self.db = db
        self.base_data_dir = Path(model_save_path).parent
        self.base_data_dir.mkdir(parents=True, exist_ok=True)

        self.model_save_path = Path(model_save_path)
        self.train_data_csv_path = Path(train_data_csv_path)
        self.train_data_excel_path = Path(train_data_excel_path)

        self.trainer = RecommendationTrainer(
            model_save_path=self.model_save_path,
            train_data_csv_path=self.train_data_csv_path,
            train_data_excel_path=self.train_data_excel_path,
        )
        self.ranking_model = None

        if self.model_save_path.exists():
            self.ranking_model = lgb.Booster(model_file=str(self.model_save_path))

    def train_init(self, num_samples: int = 12000):
        total, saved = self.trainer.run_init_pipeline(num_samples=num_samples)
        # reload model after training
        if self.model_save_path.exists():
            self.ranking_model = lgb.Booster(model_file=str(self.model_save_path))
        return total, str(saved)

    def train_init_real(self, pg_db, mongo_uri: str, mongo_db_name: str, max_samples: int = 10000):
        """
        Train model using real data from PostgreSQL + MongoDB.
        
        Args:
            pg_db: psycopg connection to PostgreSQL (Supabase)
            mongo_db: Motor async database object for MongoDB
            max_samples: Maximum number of samples to fetch
        
        Returns:
            (total_samples: int, model_path: str)
        """
        logger.info("🚀 Initiating real data model training...")
        total, saved = self.trainer.run_init_pipeline_real_data(
            pg_db=pg_db,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            max_samples=max_samples
        )
        # reload model after training
        if self.model_save_path.exists():
            self.ranking_model = lgb.Booster(model_file=str(self.model_save_path))
        logger.info(f"✅ Real data training complete: {total} samples, model saved to {saved}")
        return total, str(saved)

    async def rank(self, account_id: str, series_ids: List[str]):
        if not series_ids:
            return []

        # ensure model loaded
        if self.ranking_model is None and self.model_save_path.exists():
            self.ranking_model = lgb.Booster(model_file=str(self.model_save_path))

        # If model still missing, fallback to zeros
        model_available = self.ranking_model is not None

        # Fetch user and series documents from DB
        raw_user = await self.db.user_features.find_one({"_id": account_id})
        user_doc = self._normalize_user_doc(raw_user)

        series_cursor = self.db.series_metadata.find({"_id": {"$in": series_ids}})
        raw_series_list = await series_cursor.to_list(length=len(series_ids))

        series_map = {}
        for rs in raw_series_list:
            norm = self._normalize_series_doc(rs)
            series_map[norm["id"]] = norm

        flattened = []
        valid_series_ids = []
        for s_id in series_ids:
            s_doc = series_map.get(s_id)
            if not s_doc:
                continue
            row = self.trainer.process_and_flatten(user_doc, s_doc)
            flattened.append(row)
            valid_series_ids.append(s_id)

        if not flattened:
            return [{"seriesId": s, "score": 0.0} for s in series_ids]

        df = pd.DataFrame(flattened)
        df["user_age"] = df["user_age"].astype("category")
        df["user_gender"] = df["user_gender"].astype("category")
        df["series_content_type"] = df["series_content_type"].astype("category")

        if not model_available:
            return [{"seriesId": s, "score": 0.0} for s in series_ids]

        predicted = self.ranking_model.predict(df)

        results = [
            {"seriesId": s_id, "score": round(float(score), 4)}
            for s_id, score in zip(valid_series_ids, predicted)
        ]

        results.sort(key=lambda x: x["score"], reverse=True)

        processed = set(valid_series_ids)
        for orig in series_ids:
            if orig not in processed:
                results.append({"seriesId": orig, "score": 0.0})

        return results

    def _normalize_user_doc(self, doc: dict) -> dict:
        if not doc:
            return {
                "age": 20, "gender": "UNKNOWN", "language": "vi",
                "preferences": {"preferred_genres_by_watch_time": {}, "preferred_tags_by_clicks_last_7d": {}},
                "interactions": {"like_to_click_ratio": 0.0, "bookmark_to_click_ratio": 0.0},
                "deep_engagement": {"watch_time_last_7d": 0.0}
            }

        return {
            "age": doc.get("age", 20),
            "gender": doc.get("gender", "UNKNOWN"),
            "language": doc.get("language", "vi"),
            "preferences": doc.get("preferences", {}),
            "interactions": doc.get("interactions", {}),
            "deep_engagement": doc.get("deep_engagement", {})
        }

    def _normalize_series_doc(self, doc: dict) -> dict:
        if not doc:
            return {}

        updated_at = doc.get("released_updated_at")
        if isinstance(updated_at, datetime):
            updated_at = updated_at.isoformat()

        return {
            "id": str(doc.get("_id")),
            "contentType": doc.get("content_type", "MOVIE"),
            "category": doc.get("category", []),
            "tags": doc.get("tags", []),
            "ageRating": doc.get("age_rating", "G"),
            "language": doc.get("language", "vi"),
            "rating": doc.get("rating", 0.0),
            "releasedUpdatedAt": updated_at,
            "interactionStats": doc.get("interaction_stats", {}),
            "engagementStats": doc.get("engagement_stats", {})
        }


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

def process_series_deletion(series_id: str) -> list[str]:
    """
    Called when a Series is deleted/hidden.
    Finds its neighbors before deleting it from Milvus, returns them for cross-updating.
    """
    try:
        from app.rag.milvus_recommendation_store import get_vector_by_series_id, delete_by_series_id
        vector = get_vector_by_series_id(series_id)
        
        neighbors = []
        if vector:
            raw_similar = search_similar_series(vector, top_k=11)
            if series_id in raw_similar:
                raw_similar.remove(series_id)
            neighbors = raw_similar[:10]
        
        delete_by_series_id(series_id)
        logger.info(f"Deleted vector for series_id={series_id}. Notifying {len(neighbors)} neighbors.")
        return neighbors
    except Exception as e:
        logger.error(f"Failed to process series deletion for {series_id}: {e}")
        return []

def recalculate_series(series_id: str) -> list[str]:
    """
    Called during symmetric updates.
    Recalculates neighbors using existing vector.
    """
    try:
        from app.rag.milvus_recommendation_store import get_vector_by_series_id
        vector = get_vector_by_series_id(series_id)
        if not vector:
            return []
            
        similar_ids = search_similar_series(vector, top_k=11)
        if series_id in similar_ids:
            similar_ids.remove(series_id)
        return similar_ids[:10]
    except Exception as e:
        logger.error(f"Failed to recalculate series {series_id}: {e}")
        return []