import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta

class RecommendationTrainer:
    def __init__(self, model_save_path: str = "app/services/lgb_ranking_model.txt"):
        self.model_save_path = model_save_path

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
                "age": random.randint(12, 45),
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
                },
                "monetization": {
                    "total_spent_amount": float(random.choice([0, 0, 0, 99000, 199000, 499000])),
                    "last_purchase_time": (datetime(2026, 7, 17) - timedelta(days=random.randint(0, 60))).isoformat() if random.random() > 0.7 else None
                }
            }
            mock_users.append(user_doc)
            
        return mock_users, mock_series

    def process_and_flatten(self, user_doc, series_doc):

        # --- A. User Static & Dynamic ---
        u_age = user_doc["age"]
        u_gender = user_doc["gender"]
        
        pref = user_doc["preferences"]
        user_genres_wt = pref["preferred_genres_by_watch_time"]
        user_tags_click_7d = pref["preferred_tags_by_clicks_last_7d"]
        
        u_like_ratio = user_doc["interactions"]["like_to_click_ratio"]
        u_book_ratio = user_doc["interactions"]["bookmark_to_click_ratio"]
        u_wt_7d = user_doc["deep_engagement"]["watch_time_last_7d"]
        
        u_spent = user_doc["monetization"]["total_spent_amount"]
        last_pur = user_doc["monetization"]["last_purchase_time"]
        days_since_purchase = (datetime(2026, 7, 17) - datetime.fromisoformat(str(last_pur))).days if last_pur else -1

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
        
        age_map = {"G": 0, "PG-13": 13, "R-18": 18}
        is_age_appropriate = 1 if u_age >= age_map.get(s_age_rate, 0) else 0

        return {
            "user_age": u_age, "user_gender": u_gender, "like_to_click_ratio": u_like_ratio,
            "bookmark_to_click_ratio": u_book_ratio, "watch_time_last_7d": u_wt_7d,
            "total_spent_amount": u_spent, "days_since_last_purchase": days_since_purchase,
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

        mock_users, mock_series = self.generate_mock_documents(num_users=100, num_series=200)
        
        rows = []
        for _ in range(num_samples):
            user = random.choice(mock_users)
            series = random.choice(mock_series)
            
            features = self.process_and_flatten(user, series)
            
            # CÀI CẮM LUẬT LOGIC
            base_prob = 0.05 
            if features["is_age_appropriate"] == 0:
                base_prob = 0.0  
            else:
                base_prob += features["match_genre_wt_score"] * 0.45   
                base_prob += features["match_tag_click_7d_score"] * 0.15 
                
                # Cài cắm logic: Nếu bộ truyện có tỷ lệ share hoặc comment đột biến -> Tăng mạnh khả năng phân phối click
                if features["series_share_ratio"] > 0.015:
                    base_prob += 0.12  # Cộng 12% xác suất nếu được chia sẻ nhiều
                if features["series_comment_ratio"] > 0.03:
                    base_prob += 0.08  # Cộng 8% xác suất nếu thảo luận sôi nổi
                    
                if features["series_rating"] >= 4.5:
                    base_prob += 0.1
                if features["is_language_match"] == 1:
                    base_prob += 0.05
                    
            click_probability = min(max(base_prob, 0.0), 1.0)
            label = 1 if random.random() < click_probability else 0
            
            features["label"] = label
            rows.append(features)

        df = pd.DataFrame(rows)
        df.to_csv("app/services/train_data.csv", index=False)
        
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols]
        y = df["label"]
        
        X["user_gender"] = X["user_gender"].astype("category")
        X["series_content_type"] = X["series_content_type"].astype("category")
        
        train_dataset = lgb.Dataset(X, label=y, categorical_feature=["user_gender", "series_content_type"])
        
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
        return len(df), self.model_save_path