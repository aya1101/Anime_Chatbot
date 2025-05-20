import sys
import io
import pickle
import os

# Cấu hình stdout để sử dụng UTF-8 (nếu cần cho môi trường chạy FastAPI)
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer'): # Thường thì stderr cũng cần nếu stdout cần
             sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not reconfigure stdout/stderr to UTF-8: {e}")

from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, List
import uvicorn # Để chạy FastAPI server

# Quan trọng: Import các hàm và biến cần thiết từ recomendation.py
# Điều này cũng sẽ kích hoạt việc tải model BERT và tokenizer nếu chúng chưa được tải
from recomendation import (
    load_data_from_db,
    get_content_based_recommendations_bert,
    get_simplified_collaborative_recommendations,
    tokenizer as global_tokenizer, # đổi tên để tránh xung đột nếu có biến tokenizer cục bộ
    bert_model as global_bert_model # đổi tên để tránh xung đột
)
import pandas as pd

app = FastAPI(
    title="Anime Recommendation API",
    description="API để nhận gợi ý anime sử dụng Content-Based (BERT) và Collaborative Filtering (đơn giản hóa).",
    version="0.1.0"
)

# Biến toàn cục để lưu trữ DataFrame, sẽ được tải khi khởi động
anime_df_global: pd.DataFrame = pd.DataFrame()
movie_embeddings = None
EMBEDDINGS_FILE = "bert_embeddings.pkl"

@app.on_event("startup")
async def startup_event():
    """Sự kiện xảy ra khi FastAPI khởi động."""
    global anime_df_global, movie_embeddings
    print("FastAPI application startup...")
    
    # Kiểm tra xem BERT model và tokenizer đã được tải thành công từ recomendation.py chưa
    if global_tokenizer is None or global_bert_model is None:
        print("LỖI NGHIÊM TRỌNG: BERT tokenizer hoặc model không được tải từ recomendation.py.")
        # Bạn có thể quyết định dừng ứng dụng ở đây nếu BERT là cốt lõi
        # raise RuntimeError("BERT model/tokenizer failed to load from recomendation.py")
    else:
        print("BERT tokenizer và model đã được xác nhận tải thành công.")

    print("Đang tải dữ liệu anime từ database...")
    anime_df_global = load_data_from_db()
    if anime_df_global.empty:
        print("CẢNH BÁO: Không tải được dữ liệu anime từ database. API có thể không hoạt động đúng.")
    else:
        print(f"Đã tải thành công {len(anime_df_global)} anime vào DataFrame toàn cục.")
    
    # Tải hoặc tạo embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Đang tải embeddings từ file {EMBEDDINGS_FILE}...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            movie_embeddings = pickle.load(f)
        print("Đã tải embeddings thành công.")
    else:
        print("Không tìm thấy file embeddings, sẽ tạo mới...")
        movie_embeddings = None
    
    print("FastAPI startup complete.")

@app.get("/recommendations/", 
         response_model=Dict[str, List[str]],
         summary="Nhận gợi ý Anime",
         description="Cung cấp tên một anime và nhận danh sách gợi ý dựa trên Content-Based (BERT) và Collaborative Filtering (đơn giản hóa)."
)
async def get_recommendations(movie_title: str):
    """
    Endpoint để nhận gợi ý phim.
    - **movie_title**: Tên của bộ phim bạn muốn nhận gợi ý (bắt buộc).
    """
    if anime_df_global.empty:
        raise HTTPException(status_code=503, detail="Dữ liệu anime chưa sẵn sàng hoặc không tải được. Vui lòng thử lại sau.")

    if global_tokenizer is None or global_bert_model is None:
         raise HTTPException(status_code=503, detail="Hệ thống gợi ý Content-Based (BERT) chưa sẵn sàng do model chưa được tải.")

    if movie_title not in anime_df_global['title'].values:
        raise HTTPException(status_code=404, detail=f"Phim '{movie_title}' không được tìm thấy trong cơ sở dữ liệu.")

    print(f"Nhận yêu cầu gợi ý cho: {movie_title}")

    # Lấy gợi ý Content-Based (BERT)
    cb_recs = []
    try:
        # Sử dụng .copy() để tránh SettingWithCopyWarning trong hàm gốc nếu nó chỉnh sửa df
        cb_recs = get_content_based_recommendations_bert(movie_title, anime_df_global.copy(), top_n=5) 
    except Exception as e:
        print(f"Lỗi khi lấy gợi ý Content-Based (BERT) cho '{movie_title}': {e}")
        # Có thể không ném HTTPException ở đây để vẫn trả về gợi ý từ phương pháp kia

    # Lấy gợi ý Collaborative Filtering (Đơn giản hóa)
    cf_recs = []
    try:
        cf_recs = get_simplified_collaborative_recommendations(movie_title, anime_df_global.copy(), top_n=5)
    except Exception as e:
        print(f"Lỗi khi lấy gợi ý Collaborative Filtering (Đơn giản hóa) cho '{movie_title}': {e}")

    if not cb_recs and not cf_recs:
        # Điều này có thể xảy ra nếu phim không có đủ thông tin hoặc có lỗi trong cả hai hàm
        print(f"Không tìm thấy gợi ý nào cho '{movie_title}' từ cả hai phương pháp.")
        # Trả về dictionary rỗng thay vì lỗi 404 nếu phim tồn tại nhưng không có gợi ý

    return {
        "content_based_bert": cb_recs,
        "collaborative_simplified": cf_recs
    }

@app.get("/")
async def read_root():
    return {"message": "Chào mừng đến với API Gợi Ý Anime! Truy cập /docs để xem tài liệu API."}

# Để chạy ứng dụng này:
# 1. Mở terminal.
# 2. cd đến thư mục chứa file này.
# 3. Chạy lệnh: uvicorn test_rec:app --reload
# 4. Mở trình duyệt và truy cập: http://127.0.0.1:8000/docs

if __name__ == "__main__":
    # Cấu hình này cho phép chạy trực tiếp file python, nhưng uvicorn được khuyến nghị cho production/development
    print("Để chạy ứng dụng FastAPI này một cách chính thức, hãy sử dụng lệnh:")
    print("uvicorn test_rec:app --reload")
    print("Ứng dụng sẽ có tại: http://127.0.0.1:8000")
    # uvicorn.run(app, host="127.0.0.1", port=8000) # Bỏ comment dòng này để chạy khi thực thi file python trực tiếp
    # Tuy nhiên, --reload sẽ không hoạt động tốt ở đây, nên dùng lệnh uvicorn từ terminal. 