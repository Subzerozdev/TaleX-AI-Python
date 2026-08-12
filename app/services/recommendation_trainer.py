import random
from pathlib import Path

from pymongo import MongoClient
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from loguru import logger


class RecommendationTrainer:
    def __init__(
        self,
        model_save_path: str = "data/lgb_ranking_model.txt",
        train_data_csv_path: str = "data/train_data.csv",
        train_data_excel_path: str = "data/train_data.xlsx"
    ):
        self.model_save_path = Path(model_save_path)
        self.train_data_csv_path = Path(train_data_csv_path)
        self.train_data_excel_path = Path(train_data_excel_path)

        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_data_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_data_excel_path.parent.mkdir(parents=True, exist_ok=True)

        # Danh sách thể loại (Genres Pool) 
        self.genres_pool = [
            "Hành Động", "Viễn Tưởng", "Kinh Dị", "Chính Kịch", "Hài Hước", 
            "Lãng Mạn", "Phiêu Lưu", "Giật Gân", "Hình Sự", "Hoạt Hình", 
            "Kỳ Ảo", "Trinh Thám", "Tình Cảm", "Khoa Học Viễn Tưởng", "Tâm Lý Kinh Dị", 
            "Cổ Đại", "Văn Học Hiện Đại", "Sinh Tồn", "Kịch Tính", "Châm Biếm"
        ]
        
        # Danh sách thẻ (Tags Pool) 
        self.tags_pool = [
            "Hành Động", "Phiêu Lưu", "Hài Hước", "Chính Kịch", "Kỳ Ảo", 
            "Kinh Dị", "Trinh Thám", "Lãng Mạn", "Viễn Tưởng", "Giật Gân", 
            "Hình Sự", "Lịch Sử", "Tâm Lý", "Siêu Nhiên", "Sinh Tồn", 
            "Hồi Hộp", "Bi kịch", "Đời Thường", "Tương Lai Ảo", "Hậu Tận Thế"
        ]
        
        self.languages = ["en", "vi"]

    def generate_mock_documents(self, num_users=100, num_series=200):
        """
        BƯỚC GIẢ LẬP CƠ SỞ DỮ LIỆU: Tạo ra các mock documents dạng Dictionary
        mô phỏng chính xác các Java Class trong MongoDB.
        Lưu ý: Không bao gồm monetization vì UserFeatureDocument không theo dõi chi tiêu.
        """
        mock_users = []
        mock_series = []

        # 1. Sinh danh sách Series Documents (Metadata & Stats)
        for i in range(num_series):
            s_id = f"ser_{1000 + i}"
            cats = random.sample(self.genres_pool, k=random.randint(1, 2))
            tags = random.sample(self.tags_pool, k=random.randint(1, 3))
            
            # Giả lập chỉ số tương tác thô để tính ratio
            clicks = random.randint(100, 50000)
            likes = int(clicks * random.uniform(0.05, 0.25))
            bookmarks = int(clicks * random.uniform(0.02, 0.15))
            
            # Giả lập hành vi hiếm: Share (0.1% - 3%) và Comment (0.5% - 5%)
            shares = int(clicks * random.uniform(0.001, 0.03))
            comments = int(clicks * random.uniform(0.005, 0.05))
            
            series_doc = {
                "id": s_id,
                "contentType": random.choice(["MOVIE", "COMIC"]),
                "category": cats,
                "tags": tags,
                "ageRating": random.choice(["G", "PG-13", "R-18"]),
                "language": random.choice(self.languages),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "releasedUpdatedAt": (datetime(2026, 7, 17) - timedelta(days=random.randint(0, 180))).isoformat(),
                "interactionStats": {
                    "total_clicks": clicks,
                    "clicks_last_7d": int(clicks * 0.1),
                    "like_to_click_ratio": likes / clicks,
                    "like_to_click_ratio_last_7d": (likes / clicks) * random.uniform(0.9, 1.1),
                    "bookmark_to_click_ratio": bookmarks / clicks,
                    "bookmark_to_click_ratio_last_7d": (bookmarks / clicks) * random.uniform(0.9, 1.1),
                    "share_to_click_ratio": shares / clicks,
                    "share_to_click_ratio_last_7d": (shares / clicks) * random.uniform(0.9, 1.1),
                    "comment_to_click_ratio": comments / clicks,
                    "comment_to_click_ratio_last_7d": (comments / clicks) * random.uniform(0.9, 1.1),
                },
                "engagementStats": {
                    "watch_time_last_7d": float(clicks * random.uniform(5.0, 20.0)),
                    "watch_time_last_24h": float(clicks * random.uniform(0.5, 3.0))
                }
            }
            mock_series.append(series_doc)

        # 2. Sinh danh sách User Documents (Static & Dynamic)
        for i in range(num_users):
            u_id = f"usr_{5000 + i}"
            fav_genre = random.choice(self.genres_pool)
            fav_tag = random.choice(self.tags_pool)
            
            # Tạo map sở thích động chuẩn hóa tổng = 1.0
            genres_wt = {g: 0.01 for g in self.genres_pool}
            genres_wt[fav_genre] = 0.8  
            
            tags_clicks = {t: 0.01 for t in self.tags_pool}
            tags_clicks[fav_tag] = 0.8

            user_doc = {
                "age": random.choice(["TEEN", "MATURE"]),
                "gender": random.choice(["MALE", "FEMALE", "UNKNOWN"]),
                "language": random.choice(self.languages),
                "onboarding_movie_genres": random.sample(self.genres_pool, k=2),
                "onboarding_comic_genres": random.sample(self.genres_pool, k=2),
                "preferences": {
                    "preferred_genres_by_watch_time": genres_wt,
                    "preferred_tags_by_clicks_last_7d": tags_clicks
                },
                "interactions": {
                    "like_to_click_ratio": random.uniform(0.05, 0.3),
                    "bookmark_to_click_ratio": random.uniform(0.02, 0.2)
                },
                "deep_engagement": {
                    "watch_time_last_7d": random.uniform(60.0, 1200.0)
                }
            }
            mock_users.append(user_doc)
            
        return mock_users, mock_series

    def process_and_flatten(self, user_doc, series_doc):
        """Extract and flatten 26 features from user and series documents for LightGBM."""

        # --- A. User Static & Dynamic ---
        u_age = user_doc["age"]
        u_gender = user_doc["gender"]
        
        pref = user_doc["preferences"]
        user_genres_wt = pref["preferred_genres_by_watch_time"]
        user_tags_click_7d = pref["preferred_tags_by_clicks_last_7d"]
        
        u_like_ratio = user_doc["interactions"]["like_to_click_ratio"]
        u_book_ratio = user_doc["interactions"]["bookmark_to_click_ratio"]
        u_wt_7d = user_doc["deep_engagement"]["watch_time_last_7d"]

        # --- B. Series Static & Dynamic ---
        s_type = series_doc["contentType"]
        s_rating = series_doc["rating"]
        s_age_rate = series_doc["ageRating"]
        
        updated_at = series_doc["releasedUpdatedAt"]
        days_since_update = (datetime(2026, 7, 17) - datetime.fromisoformat(str(updated_at))).days
        
        s_inter = series_doc["interactionStats"]
        s_clicks = s_inter["total_clicks"]
        s_clicks_7d = s_inter["clicks_last_7d"]
        
        s_like_ratio = s_inter["like_to_click_ratio"]
        s_like_ratio_7d = s_inter["like_to_click_ratio_last_7d"]
        s_book_ratio = s_inter["bookmark_to_click_ratio"]
        s_book_ratio_7d = s_inter["bookmark_to_click_ratio_last_7d"]
        
        s_share_ratio = s_inter["share_to_click_ratio"]
        s_share_ratio_7d = s_inter["share_to_click_ratio_last_7d"]
        s_comment_ratio = s_inter["comment_to_click_ratio"]
        s_comment_ratio_7d = s_inter["comment_to_click_ratio_last_7d"]
        
        s_eng = series_doc["engagementStats"]
        s_wt_7d = s_eng["watch_time_last_7d"]
        s_wt_24h = s_eng["watch_time_last_24h"]

        # --- C. Cross Features ---
        match_genre_wt_score = sum([user_genres_wt.get(cat, 0.0) for cat in series_doc["category"]])
        match_tag_click_7d_score = sum([user_tags_click_7d.get(tag, 0.0) for tag in series_doc["tags"]])
        is_language_match = 1 if user_doc["language"] == series_doc["language"] else 0
        
        # Age appropriateness: u_age is string (TEEN or MATURE)
        # G and PG-13 are OK for all ages, R-18 only for MATURE
        is_age_appropriate = 1 if s_age_rate != "R-18" or u_age == "MATURE" else 0

        return {
            "user_age": u_age, "user_gender": u_gender, "like_to_click_ratio": u_like_ratio,
            "bookmark_to_click_ratio": u_book_ratio, "watch_time_last_7d": u_wt_7d,
            "series_content_type": s_type, "series_rating": s_rating, "days_since_last_update": days_since_update,
            "series_wt_7d": s_wt_7d, "series_wt_24h": s_wt_24h, 
            "series_like_ratio": s_like_ratio, "series_like_ratio_7d": s_like_ratio_7d, 
            "series_book_ratio": s_book_ratio, "series_book_ratio_7d": s_book_ratio_7d,
            "series_share_ratio": s_share_ratio, "series_share_ratio_7d": s_share_ratio_7d,
            "series_comment_ratio": s_comment_ratio, "series_comment_ratio_7d": s_comment_ratio_7d,
            "log_series_clicks": np.log10(s_clicks + 1), "log_series_clicks_7d": np.log10(s_clicks_7d + 1),
            "match_genre_wt_score": match_genre_wt_score, "match_tag_click_7d_score": match_tag_click_7d_score,
            "is_language_match": is_language_match, "is_age_appropriate": is_age_appropriate
        }

    def run_init_pipeline(self, num_samples=12000):
        """Mock data pipeline: generate synthetic data, train LightGBM model, export to CSV/Excel."""

        mock_users, mock_series = self.generate_mock_documents(num_users=100, num_series=200)
        
        rows = []
        for _ in range(num_samples):
            user = random.choice(mock_users)
            series = random.choice(mock_series)
            
            features = self.process_and_flatten(user, series)
            
            # Embedded logic: determine if user would interact based on features
            base_prob = 0.05 
            if features["is_age_appropriate"] == 0:
                base_prob = 0.0  
            else:
                base_prob += features["match_genre_wt_score"] * 0.45   
                base_prob += features["match_tag_click_7d_score"] * 0.15 
                
                if features["series_share_ratio"] > 0.015:
                    base_prob += 0.12
                if features["series_comment_ratio"] > 0.03:
                    base_prob += 0.08
                    
                if features["series_rating"] >= 4.5:
                    base_prob += 0.1
                if features["is_language_match"] == 1:
                    base_prob += 0.05
                    
            click_probability = min(max(base_prob, 0.0), 1.0)
            label = 1 if random.random() < click_probability else 0
            
            features["label"] = label
            rows.append(features)

        df = pd.DataFrame(rows)
        df.to_csv(self.train_data_csv_path, index=False)
        df.to_excel(self.train_data_excel_path, index=False)
        
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols]
        y = df["label"]

        X["user_age"] = X["user_age"].astype("category")
        X["user_gender"] = X["user_gender"].astype("category")
        X["series_content_type"] = X["series_content_type"].astype("category")
        
        train_dataset = lgb.Dataset(X, label=y, categorical_feature=["user_age", "user_gender", "series_content_type"])
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(params, train_dataset, num_boost_round=120)
        model.save_model(self.model_save_path)
        return len(df), str(self.model_save_path)

    def run_init_pipeline_real_data(self, pg_db, mongo_uri: str, mongo_db_name: str, max_samples: int = 10000):
        """
        Real data pipeline: Fetch training data from PostgreSQL + MongoDB.
        
        PostgreSQL (Supabase):
            - Queries AccountImpression table for (account_id, series_id, is_watched, is_interacted)
            
        MongoDB (Atlas):
            - Fetches UserFeatureDocument from user_features collection
            - Fetches SeriesMetadata from series_metadata collection
            
        Logs all query results for debugging.
        """
        """Real data pipeline using PyMongo sync client and batch fetching."""
        logger.info("=" * 70)
        logger.info("🚀 Starting REAL DATA PIPELINE for recommendation model training")
        logger.info("=" * 70)

        try:
            # Step 1: Lấy danh sách impressions từ PostgreSQL
            logger.info(f"📊 Querying PostgreSQL: SELECT up to {max_samples} from account_impressions...")
            impressions_query = """
                SELECT ai.account_id, ai.series_id, ai.is_watched, ai.is_interacted
                FROM account_impressions ai
                LIMIT %s
            """
            
            cursor = pg_db.cursor()
            cursor.execute(impressions_query, (max_samples,))
            impressions = cursor.fetchall()
            cursor.close()
            
            impression_count = len(impressions)
            logger.info(f"✅ Fetched {impression_count} AccountImpression records from PostgreSQL")
            
            if impression_count == 0:
                logger.error("❌ No AccountImpression records found! Returning mock data instead.")
                return self.run_init_pipeline(num_samples=12000)

            # Step 2: Tạo kết nối PyMongo đồng bộ & Batch Fetch dữ liệu từ MongoDB
            mongo_client = MongoClient(mongo_uri)
            mongo_db = mongo_client[mongo_db_name]

            unique_account_ids = list(set(str(acc_id) for acc_id, _, _, _ in impressions if acc_id))
            unique_series_ids = list(set(str(s_id) for _, s_id, _, _ in impressions if s_id))

            logger.info(f"🔄 Batch querying MongoDB for {len(unique_account_ids)} users and {len(unique_series_ids)} series...")

            # Lấy toàn bộ User Documents liên quan trong 1 query
            user_docs_raw = list(mongo_db.user_features.find({"_id": {"$in": unique_account_ids}}))
            user_map = {str(doc["_id"]): doc for doc in user_docs_raw}
            logger.info(f"✅ Batch loaded {len(user_map)} UserFeatureDocuments from MongoDB")

            # Lấy toàn bộ Series Metadata liên quan trong 1 query
            series_docs_raw = list(mongo_db.series_metadata.find({"_id": {"$in": unique_series_ids}}))
            series_map = {str(doc["_id"]): doc for doc in series_docs_raw}
            logger.info(f"✅ Batch loaded {len(series_map)} SeriesMetadata documents from MongoDB")

            mongo_client.close()

            # Step 3: Ghép nối dữ liệu trên RAM
            rows = []
            processed = 0
            failed = 0

            for account_id, series_id, is_watched, is_interacted in impressions:
                try:
                    label = 1 if is_watched or is_interacted else 0
                    
                    user_doc = user_map.get(str(account_id))
                    if not user_doc:
                        failed += 1
                        continue

                    series_doc = series_map.get(str(series_id))
                    if not series_doc:
                        failed += 1
                        continue

                    normalized_user = self._normalize_user_doc_real(user_doc)
                    normalized_series = self._normalize_series_doc_real(series_doc)

                    features = self._process_and_flatten_real(normalized_user, normalized_series)
                    features["label"] = label

                    rows.append(features)
                    processed += 1

                except Exception as e:
                    logger.error(f"❌ Error processing impression: {str(e)}")
                    failed += 1
                    continue

            logger.info(f"✅ Successfully processed {processed} impressions (Failed: {failed})")

            if len(rows) == 0:
                logger.error("❌ No valid training rows generated! Returning mock data instead.")
                return self.run_init_pipeline(num_samples=12000)

            # Step 4: Tạo DataFrame và huấn luyện LightGBM
            df = pd.DataFrame(rows)
            df.to_csv(self.train_data_csv_path, index=False)
            df.to_excel(self.train_data_excel_path, index=False)

            feature_cols = [c for c in df.columns if c != "label"]
            X = df[feature_cols]
            y = df["label"]

            # Chuyển đổi định dạng category cho LightGBM
            X["user_age"] = X["user_age"].astype("category")
            X["user_gender"] = X["user_gender"].astype("category")
            X["series_content_type"] = X["series_content_type"].astype("category")

            train_dataset = lgb.Dataset(
                X, 
                label=y, 
                categorical_feature=["user_age", "user_gender", "series_content_type"]
            )

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1,
                'seed': 42
            }

            model = lgb.train(params, train_dataset, num_boost_round=120)
            model.save_model(self.model_save_path)

            logger.info(f"✅ Model trained on {len(rows)} real samples and saved to: {self.model_save_path}")
            return len(df), str(self.model_save_path)

        except Exception as e:
            logger.error(f"💥 FATAL ERROR in real data pipeline: {str(e)}", exc_info=True)
            return self.run_init_pipeline(num_samples=12000)

    def _normalize_user_doc_real(self, doc: dict) -> dict:
        """Normalize real UserFeatureDocument from MongoDB to standard dict format.
        Age is converted to string category: TEEN (11-18) or MATURE (18+)
        Default is TEEN if null or empty.
        """
        if not doc:
            return {
                "age": "TEEN", "gender": "UNKNOWN", "language": "vi",
                "preferences": {"preferred_genres_by_watch_time": {}, "preferred_tags_by_clicks_last_7d": {}},
                "interactions": {"like_to_click_ratio": 0.0, "bookmark_to_click_ratio": 0.0},
                "deep_engagement": {"watch_time_last_7d": 0.0}
            }
        
        # Convert age to TEEN or MATURE category
        age_value = doc.get("age", "")
        if not age_value or (isinstance(age_value, str) and age_value.strip() == ""):
            age_category = "TEEN"
        else:
            try:
                age_int = int(age_value) if isinstance(age_value, str) else age_value
                age_category = "MATURE" if age_int >= 18 else "TEEN"
            except (ValueError, TypeError):
                age_category = "TEEN"
        
        return {
            "age": age_category,
            "gender": doc.get("gender", "UNKNOWN"),
            "language": doc.get("language", "vi"),
            "preferences": doc.get("preferences", {}),
            "interactions": doc.get("interactions", {}),
            "deep_engagement": doc.get("deep_engagement", {})
        }

    def _normalize_series_doc_real(self, doc: dict) -> dict:
        """Normalize real SeriesMetadata from MongoDB to standard dict format."""
        if not doc:
            return {}
        
        updated_at = doc.get("released_updated_at")
        if isinstance(updated_at, str):
            updated_at = updated_at
        elif hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()
        else:
            updated_at = datetime.now().isoformat()
        
        return {
            "id": str(doc.get("_id", "")),
            "contentType": doc.get("content_type", doc.get("contentType", "MOVIE")),
            "category": doc.get("category", []),
            "tags": doc.get("tags", []),
            "ageRating": doc.get("age_rating", doc.get("ageRating", "G")),
            "language": doc.get("language", "vi"),
            "rating": float(doc.get("rating", 0.0)),
            "releasedUpdatedAt": updated_at,
            "interactionStats": doc.get("interaction_stats", doc.get("interactionStats", {})),
            "engagementStats": doc.get("engagement_stats", doc.get("engagementStats", {}))
        }

    def _process_and_flatten_real(self, user_doc, series_doc):
        """
        Process and flatten features from real data (without monetization).
        Returns dict of 26 features for LightGBM model.
        Age is expected as string category: TEEN or MATURE.
        """
        # --- A. User Static & Dynamic ---
        u_age = user_doc.get("age", "TEEN")  # Default to TEEN
        u_gender = user_doc.get("gender", "UNKNOWN")
        
        pref = user_doc.get("preferences", {})
        user_genres_wt = pref.get("preferred_genres_by_watch_time", {})
        user_tags_click_7d = pref.get("preferred_tags_by_clicks_last_7d", {})
        
        interactions = user_doc.get("interactions", {})
        u_like_ratio = interactions.get("like_to_click_ratio", 0.0)
        u_book_ratio = interactions.get("bookmark_to_click_ratio", 0.0)
        
        deep_eng = user_doc.get("deep_engagement", {})
        u_wt_7d = deep_eng.get("watch_time_last_7d", 0.0)

        # --- B. Series Static & Dynamic ---
        s_type = series_doc.get("contentType", "MOVIE")
        s_rating = series_doc.get("rating", 0.0)
        s_age_rate = series_doc.get("ageRating", "G")
        
        updated_at = series_doc.get("releasedUpdatedAt", datetime.now().isoformat())
        try:
            days_since_update = (datetime(2026, 7, 17) - datetime.fromisoformat(str(updated_at))).days
        except:
            days_since_update = 0
        
        inter_stats = series_doc.get("interactionStats", {})
        s_clicks = inter_stats.get("total_clicks", 1)
        s_clicks_7d = inter_stats.get("clicks_last_7d", 0)
        
        s_like_ratio = inter_stats.get("like_to_click_ratio", 0.0)
        s_like_ratio_7d = inter_stats.get("like_to_click_ratio_last_7d", 0.0)
        s_book_ratio = inter_stats.get("bookmark_to_click_ratio", 0.0)
        s_book_ratio_7d = inter_stats.get("bookmark_to_click_ratio_last_7d", 0.0)
        
        s_share_ratio = inter_stats.get("share_to_click_ratio", 0.0)
        s_share_ratio_7d = inter_stats.get("share_to_click_ratio_last_7d", 0.0)
        s_comment_ratio = inter_stats.get("comment_to_click_ratio", 0.0)
        s_comment_ratio_7d = inter_stats.get("comment_to_click_ratio_last_7d", 0.0)
        
        eng_stats = series_doc.get("engagementStats", {})
        s_wt_7d = eng_stats.get("watch_time_last_7d", 0.0)
        s_wt_24h = eng_stats.get("watch_time_last_24h", 0.0)

        # --- C. Cross Features ---
        match_genre_wt_score = sum([user_genres_wt.get(cat, 0.0) for cat in series_doc.get("category", [])])
        match_tag_click_7d_score = sum([user_tags_click_7d.get(tag, 0.0) for tag in series_doc.get("tags", [])])
        is_language_match = 1 if user_doc.get("language") == series_doc.get("language") else 0
        
        # Age appropriateness: u_age is string (TEEN or MATURE)
        # G and PG-13 are OK for all ages, R-18 only for MATURE
        is_age_appropriate = 1 if s_age_rate != "R-18" or u_age == "MATURE" else 0

        return {
            "user_age": u_age, 
            "user_gender": u_gender, 
            "like_to_click_ratio": u_like_ratio,
            "bookmark_to_click_ratio": u_book_ratio, 
            "watch_time_last_7d": u_wt_7d,
            "series_content_type": s_type, 
            "series_rating": s_rating, 
            "days_since_last_update": days_since_update,
            "series_wt_7d": s_wt_7d, 
            "series_wt_24h": s_wt_24h, 
            "series_like_ratio": s_like_ratio, 
            "series_like_ratio_7d": s_like_ratio_7d, 
            "series_book_ratio": s_book_ratio, 
            "series_book_ratio_7d": s_book_ratio_7d,
            "series_share_ratio": s_share_ratio, 
            "series_share_ratio_7d": s_share_ratio_7d,
            "series_comment_ratio": s_comment_ratio, 
            "series_comment_ratio_7d": s_comment_ratio_7d,
            "log_series_clicks": np.log10(s_clicks + 1), 
            "log_series_clicks_7d": np.log10(s_clicks_7d + 1),
            "match_genre_wt_score": match_genre_wt_score, 
            "match_tag_click_7d_score": match_tag_click_7d_score,
            "is_language_match": is_language_match, 
            "is_age_appropriate": is_age_appropriate
        }
