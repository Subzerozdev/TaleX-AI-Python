from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import psycopg2
import os

from app.core.config import settings
from app.services.recommendation_service import RecommendationService

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
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = DATA_DIR / "lgb_ranking_model.txt"
TRAIN_DATA_EXCEL_PATH = DATA_DIR / "train_data.xlsx"

# Service instance encapsulates training and ranking logic
service = RecommendationService(
    db=db,
    model_save_path=str(MODEL_PATH),
    train_data_csv_path=str(DATA_DIR / "train_data.csv"),
    train_data_excel_path=str(TRAIN_DATA_EXCEL_PATH),
)


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
# Normalization and DB logic moved into RecommendationService


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
        total_samples, saved_path = service.train_init(num_samples=12000)
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


@router.post("/train-init-real", response_model=TrainInitResponse, status_code=status.HTTP_201_CREATED)
def trigger_train_init_real(token: str = None, max_samples: int = 10000):
    """
    Khởi tạo bộ não từ DỮ LIỆU THỰC tế từ PostgreSQL (Supabase) + MongoDB (Atlas).
    
    Quá trình:
    1. Kết nối tới PostgreSQL và lấy danh sách AccountImpression
    2. Với mỗi impression, lấy UserFeatureDocument + SeriesMetadata từ MongoDB
    3. Trích xuất 26 features, log chi tiết cho debugging
    4. Huấn luyện LightGBM model trên dữ liệu thực
    5. Lưu model và export train_data.csv + train_data.xlsx
    
    Nếu không tìm thấy dữ liệu thực, sẽ fallback về mock data pipeline.
    """
    if token and token != "talex_secret_demo_2026":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Mã token kích hoạt không hợp lệ!"
        )
    
    try:
        # Tạo kết nối PostgreSQL từ .env
        pg_host = os.getenv("DB_HOST", "aws-1-ap-southeast-2.pooler.supabase.com")
        pg_port = os.getenv("DB_PORT", "6543")
        pg_database = os.getenv("DB_NAME", "postgres")
        pg_user = os.getenv("DB_USERNAME", "")
        pg_password = os.getenv("DB_PASSWORD", "")
        
        if not pg_user or not pg_password:
            raise ValueError("PostgreSQL credentials missing in .env (DB_USERNAME, DB_PASSWORD)")
        
        pg_conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        
        # Gọi service.train_init_real với PostgreSQL + MongoDB connections
        total_samples, saved_path = service.train_init_real(
            pg_db=pg_conn, 
            mongo_uri=settings.MONGO_URI, 
            mongo_db_name=settings.MONGO_DB_NAME, 
            max_samples=max_samples
        )
        
        # Đóng kết nối PostgreSQL
        pg_conn.close()
        
        return TrainInitResponse(
            status="SUCCESS",
            message=f"Đã khởi tạo thành công bộ não từ {total_samples} dòng dữ liệu THỰC (PostgreSQL + MongoDB)!",
            total_samples_generated=total_samples,
            model_saved_at=saved_path
        )
        
    except psycopg2.OperationalError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không kết nối được tới PostgreSQL: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi cấu hình: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi huấn luyện với dữ liệu thực: {str(e)}"
        )


@router.post("/rank", response_model=List[RankResultItem])
async def rank_candidates(payload: RankRequest = Body(...)):
    """
    API Xếp Hạng Tinh: Lấy dữ liệu thực từ MongoDB Atlas, phẳng hóa đặc trưng
    và sử dụng LightGBM để tính toán điểm số xác suất phản hồi cụ thể.
    """
    if not payload.seriesIds:
        return []

    try:
        ranked = await service.rank(payload.accountId, payload.seriesIds)
        return [RankResultItem(**r) for r in ranked]
    except Exception as e:
        print(f"[❌ AI REALTIME ERROR] Sự cố xảy ra: {str(e)}")
        return [RankResultItem(seriesId=s_id, score=0.0) for s_id in payload.seriesIds]


@router.get("/train-data/download")
def download_train_data():
    """Tải file tập dữ liệu huấn luyện dạng Excel từ server."""
    excel_path = TRAIN_DATA_EXCEL_PATH
    if not excel_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy file train_data.xlsx. Vui lòng chạy /train-init trước."
        )
    return FileResponse(
        path=str(excel_path),
        filename=excel_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )