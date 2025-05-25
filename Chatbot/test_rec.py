import sys
import io
import pickle
import os
from contextlib import asynccontextmanager

# Cấu hình stdout để sử dụng UTF-8
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not reconfigure stdout/stderr to UTF-8: {e}")

from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, List
import uvicorn
import pandas as pd

# Import các hàm từ recomendation.py
from recomendation import (
    load_data_from_db,
    get_content_based_recommendations_tfidf,
    get_simplified_collaborative_recommendations
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler cho FastAPI."""
    # Startup
    global anime_df_global
    print("FastAPI application startup...")
    
    print("Đang tải dữ liệu anime từ database...")
    anime_df_global = load_data_from_db()
    if anime_df_global.empty:
        print("CẢNH BÁO: Không tải được dữ liệu anime từ database. API có thể không hoạt động đúng.")
    else:
        print(f"Đã tải thành công {len(anime_df_global)} anime vào DataFrame toàn cục.")
    
    yield
    # Shutdown
    print("FastAPI application shutdown...")

app = FastAPI(
    title="Anime Recommendation API",
    description="API để nhận gợi ý anime sử dụng Content-Based (TF-IDF) và Collaborative Filtering (đơn giản hóa).",
    version="0.1.0",
    lifespan=lifespan
)

# Biến toàn cục để lưu trữ DataFrame
anime_df_global: pd.DataFrame = pd.DataFrame()

@app.get("/recommendations/", 
         response_model=Dict[str, List[Dict[str, str]]],
         summary="Nhận gợi ý Anime",
         description="Cung cấp tên một anime và nhận danh sách gợi ý dựa trên Content-Based (TF-IDF) và Collaborative Filtering (đơn giản hóa)."
)
async def get_recommendations(movie_title: str):
    """
    Endpoint để nhận gợi ý phim.
    - **movie_title**: Tên của bộ phim bạn muốn nhận gợi ý (bắt buộc).
    """
    if anime_df_global.empty:
        raise HTTPException(status_code=503, detail="Dữ liệu anime chưa sẵn sàng hoặc không tải được. Vui lòng thử lại sau.")

    if movie_title not in anime_df_global['title'].values:
        raise HTTPException(status_code=404, detail=f"Phim '{movie_title}' không được tìm thấy trong cơ sở dữ liệu.")

    print(f"Nhận yêu cầu gợi ý cho: {movie_title}")

    # Lấy gợi ý Content-Based (TF-IDF)
    cb_recs = []
    try:
        cb_recs = get_content_based_recommendations_tfidf(movie_title, anime_df_global.copy(), top_n=5)
    except Exception as e:
        print(f"Lỗi khi lấy gợi ý Content-Based (TF-IDF) cho '{movie_title}': {e}")

    # Lấy gợi ý Collaborative Filtering (Đơn giản hóa)
    cf_recs = []
    try:
        cf_recs = get_simplified_collaborative_recommendations(movie_title, anime_df_global.copy(), top_n=5)
    except Exception as e:
        print(f"Lỗi khi lấy gợi ý Collaborative Filtering (Đơn giản hóa) cho '{movie_title}': {e}")

    if not cb_recs and not cf_recs:
        print(f"Không tìm thấy gợi ý nào cho '{movie_title}' từ cả hai phương pháp.")

    return {
        "content_based_tfidf": cb_recs,
        "collaborative_simplified": cf_recs
    }

@app.get("/")
async def read_root():
    return {"message": "Chào mừng đến với API Gợi Ý Anime! Truy cập /docs để xem tài liệu API."}

if __name__ == "__main__":
    print("Để chạy ứng dụng FastAPI này một cách chính thức, hãy sử dụng lệnh:")
    print("uvicorn test_rec:app --reload")
    print("Ứng dụng sẽ có tại: http://127.0.0.1:8000") 