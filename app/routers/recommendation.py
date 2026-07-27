import os
from datetime import datetime
from typing import List
import lightgbm as lgb
import pandas as pd
from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient # Thư viện kết nối MongoDB Async

from app.core.config import settings
from app.services.recommendation_trainer import RecommendationTrainer

router = APIRouter(
    prefix="/api/v1/recommendations",
    tags=["Recommendation System"]
)

# =====================================================================
# CẤU HÌNH KẾT NỐI MONGODB REAL-TIME (CLOUD ATLAS)
# =====================================================================
MONGO_URI = settings.MONGO_URI
DB_NAME = settings.MONGO_DB_NAME

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# =====================================================================
# KHỞI TẠO & LOAD BỘ NÃO AI LÊN RAM NGAY KHI START SERVER
# =====================================================================
MODEL_PATH = "app/services/lgb_ranking_model.txt"
trainer = RecommendationTrainer(model_save_path=MODEL_PATH)

if os.path.exists(MODEL_PATH):
    print(f"[AI ENGINE] Đang nạp mô hình LightGBM từ {MODEL_PATH} lên RAM...")
    ranking_model = lgb.Booster(model_file=MODEL_PATH)
else:
    print(f"[⚠️ CẢNH BÁO] Không tìm thấy file {MODEL_PATH}! Vui lòng chạy endpoint /train-init trước.")
    ranking_model = None


# =====================================================================
# DEFINITION DATA TRANSFER OBJECTS (DTOs)
# =====================================================================
class TrainInitResponse(BaseModel):
    status: str
    message: str
    total_samples_generated: int
    model_saved_at: str

class RankRequest(BaseModel):
    accountId: str
    seriesIds: List[str]

# DTO mới định nghĩa cấu trúc trả về gồm cả ID và Điểm số AI chấm
class RankResultItem(BaseModel):
    seriesId: str
    score: float


# =====================================================================
# HELPER FUNCTIONS: CHUẨN HÓA DỮ LIỆU TỪ MONGODB (DATA NORMALIZATION)
# =====================================================================
def normalize_user_doc(doc: dict) -> dict:
    """Đảm bảo cấu trúc map dữ liệu User luôn an toàn trước khi nạp vào Trainer"""
    if not doc:
        return {
            "age": 20, "gender": "UNKNOWN", "language": "vi",
            "preferences": {"preferred_genres_by_watch_time": {}, "preferred_tags_by_clicks_last_7d": {}},
            "interactions": {"like_to_click_ratio": 0.0, "bookmark_to_click_ratio": 0.0},
            "deep_engagement": {"watch_time_last_7d": 0.0},
            "monetization": {"total_spent_amount": 0.0, "last_purchase_time": None}
        }
    
    # Xử lý chuẩn hóa trường thời gian BSON Date nếu có sang định dạng chuỗi ISO
    monetization = doc.get("monetization", {})
    last_pur = monetization.get("last_purchase_time")
    if isinstance(last_pur, datetime):
        monetization["last_purchase_time"] = last_pur.isoformat()

    return {
        "age": doc.get("age", 20),
        "gender": doc.get("gender", "UNKNOWN"),
        "language": doc.get("language", "vi"),
        "preferences": doc.get("preferences", {}),
        "interactions": doc.get("interactions", {}),
        "deep_engagement": doc.get("deep_engagement", {}),
        "monetization": monetization
    }

def normalize_series_doc(doc: dict) -> dict:
    """Chuyển đổi các snake_case từ Mongo BSON về camelCase mà hàm phẳng hóa yêu cầu"""
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


# =====================================================================
# CORE API ENDPOINTS
# =====================================================================
@router.post("/train-init", response_model=TrainInitResponse, status_code=status.HTTP_201_CREATED)
def trigger_train_init(token: str = None):
    """Chỉ chạy 1 lần trước demo hoặc khi cần reset bộ não cho hệ thống."""
    if token and token != "talex_secret_demo_2026":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Mã token kích hoạt không hợp lệ!"
        )
        
    try:
        trainer_init = RecommendationTrainer()
        total_samples, saved_path = trainer_init.run_init_pipeline(num_samples=12000)
        return TrainInitResponse(
            status="SUCCESS",
            message="Đã khởi tạo thành công kho dữ liệu mẫu và đồng bộ bộ não LightGBM!",
            total_samples_generated=total_samples,
            model_saved_at=saved_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi sinh dữ liệu: {str(e)}"
        )


@router.post("/rank", response_model=List[RankResultItem])
async def rank_candidates(payload: RankRequest = Body(...)):
    """
    API Xếp Hạng Tinh: Lấy dữ liệu thực từ MongoDB Atlas, phẳng hóa đặc trưng
    và sử dụng LightGBM để tính toán điểm số xác suất phản hồi cụ thể.
    """
    global ranking_model
    
    # Phòng thủ nếu chưa có file model vật lý
    if ranking_model is None:
        if os.path.exists(MODEL_PATH):
            ranking_model = lgb.Booster(model_file=MODEL_PATH)
        else:
            # Nếu chưa có model, trả về điểm mặc định 0.0 cho tất cả ứng viên
            return [RankResultItem(seriesId=s_id, score=0.0) for s_id in payload.seriesIds]

    if not payload.seriesIds:
        return []

    try:
        # -----------------------------------------------------------------
        # BƯỚC 1: TRUY VẤN MONGODB ATLAS THẬT (Dựa trên Java Entity mappings)
        # -----------------------------------------------------------------
        # Lấy document từ collection 'user_features' qua khoá chính '_id'[cite: 1]
        raw_user = await db.user_features.find_one({"_id": payload.accountId})
        user_doc = normalize_user_doc(raw_user)

        # Lấy danh sách tài liệu từ collection 'series_metadata' bằng toán tử $in[cite: 6]
        series_cursor = db.series_metadata.find({"_id": {"$in": payload.seriesIds}})
        raw_series_list = await series_cursor.to_list(length=len(payload.seriesIds))

        # Đưa vào Map để tìm kiếm nhanh theo ID gốc O(1)
        series_map = {}
        for rs in raw_series_list:
            norm_s = normalize_series_doc(rs)
            series_map[norm_s["id"]] = norm_s

        # -----------------------------------------------------------------
        # BƯỚC 2: PHẲNG HÓA VÀ TRÍCH XUẤT ĐẶC TRƯNG CHÉO (FEATURE ENGINEERING)
        # -----------------------------------------------------------------
        flattened_features = []
        valid_series_ids = []

        for s_id in payload.seriesIds:
            s_doc = series_map.get(s_id)
            if not s_doc:
                continue # Bỏ qua nếu ID không thực sự tồn tại trong Database
            
            # Đưa qua hàm xử lý logic chung (bao gồm tương tác, share, comment...)
            row_feat = trainer.process_and_flatten(user_doc, s_doc)
            flattened_features.append(row_feat)
            valid_series_ids.append(s_id)

        # Nếu không trích xuất được bất kỳ dữ liệu nào từ DB, trả về điểm 0.0
        if not flattened_features:
            return [RankResultItem(seriesId=s_id, score=0.0) for s_id in payload.seriesIds]

        # Khởi tạo ma trận dữ liệu dự đoán
        df_inference = pd.DataFrame(flattened_features)
        df_inference["user_gender"] = df_inference["user_gender"].astype("category")
        df_inference["series_content_type"] = df_inference["series_content_type"].astype("category")

        # -----------------------------------------------------------------
        # BƯỚC 3: AI INFERENCE CHẤM ĐIỂM VÀ ĐÓNG GÓI KẾT QUẢ CÓ ĐIỂM SỐ
        # -----------------------------------------------------------------
        predicted_scores = ranking_model.predict(df_inference)

        # Gộp ID cùng với điểm số tương ứng và làm tròn 4 chữ số thập phân
        candidate_results = [
            RankResultItem(seriesId=s_id, score=round(float(score), 4))
            for s_id, score in zip(valid_series_ids, predicted_scores)
        ]

        # Sắp xếp danh sách kết quả dựa trên trường 'score' từ cao xuống thấp
        candidate_results.sort(key=lambda x: x.score, reverse=True)

        # Trường hợp có series ứng viên nào bị thiếu do lệch DB, tự động bổ sung xuống cuối với điểm 0.0
        processed_ids = set(valid_series_ids)
        for original_id in payload.seriesIds:
            if original_id not in processed_ids:
                candidate_results.append(RankResultItem(seriesId=original_id, score=0.0))

        print(f"[AI MONGO REALTIME SUCCESS] Đã chấm điểm xong cho User {payload.accountId}")
        return candidate_results

    except Exception as e:
        print(f"[❌ AI REALTIME ERROR] Sự cố xảy ra: {str(e)}")
        # Fallback khi lỗi hệ thống: Trả về danh sách thô kèm điểm số 0.0
        return [RankResultItem(seriesId=s_id, score=0.0) for s_id in payload.seriesIds]