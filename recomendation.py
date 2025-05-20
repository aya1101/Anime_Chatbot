import sys
import io
import pickle

# Cấu hình stdout để sử dụng UTF-8, giải quyết vấn đề UnicodeEncodeError trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity # Changed from linear_kernel
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Cấu hình kết nối MySQL (Giống với import_to_database.py)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Thay bằng username MySQL của bạn
    'password': '123456',  # Thay bằng password MySQL của bạn
    'database': 'anime_db' # Tên database bạn đã tạo
}
TABLE_NAME = 'anime'

# Tải mô hình BERT và tokenizer (nên tải một lần)
# Sử dụng bert-base-uncased cho tiếng Anh. Nếu có tiếng Việt, cân nhắc 'bert-base-multilingual-cased'
TOKENIZER_NAME = 'bert-base-uncased'
MODEL_NAME = 'bert-base-uncased'
tokenizer = None
bert_model = None
try:
    print(f"Đang tải tokenizer: {TOKENIZER_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Đang tải model: {MODEL_NAME}...")
    bert_model = BertModel.from_pretrained(MODEL_NAME)
    print("BERT model và tokenizer đã tải xong.")
    # Đặt model ở chế độ eval nếu bạn không fine-tune
    bert_model.eval()
except Exception as e:
    print(f"Lỗi khi tải BERT model/tokenizer: {e}")
    print("Vui lòng đảm bảo bạn đã cài đặt thư viện 'transformers' và 'torch', và có kết nối internet để tải model lần đầu.")
    # Script sẽ tiếp tục nhưng các hàm dựa trên BERT sẽ không hoạt động

# Biến toàn cục để lưu trữ embeddings
movie_embeddings = None

def load_data_from_db():
    """Tải dữ liệu anime từ MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        query = """
        SELECT title, genres, description 
        FROM anime
        """
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Đã tải {len(df)} phim từ database.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu từ database: {str(e)}")
        return pd.DataFrame()

def get_bert_embedding(text, model, tokenizer_instance):
    """Tạo embedding cho một đoạn văn bản sử dụng BERT."""
    if model is None or tokenizer_instance is None:
        print("BERT model hoặc tokenizer chưa được tải. Không thể tạo embedding.")
        return np.zeros(768) 
        
    if text is None or not isinstance(text, str) or not text.strip():
        return np.zeros(model.config.hidden_size) 
    try:
        inputs = tokenizer_instance(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad(): 
            outputs = model(**inputs)
        return outputs.last_hidden_state[:,0,:].squeeze().cpu().numpy()
    except Exception as e:
        print(f"Lỗi khi tạo BERT embedding cho text '{text[:50]}...': {e}")
        return np.zeros(model.config.hidden_size)

def get_content_based_recommendations_bert(movie_title, n_recommendations=5):
    """
    Gợi ý phim dựa trên nội dung sử dụng BERT embeddings.
    
    Args:
        movie_title (str): Tên phim cần gợi ý
        n_recommendations (int): Số lượng phim gợi ý
        
    Returns:
        list: Danh sách các phim được gợi ý
    """
    global movie_embeddings
    
    # Kiểm tra xem phim có trong dataset không
    if movie_title not in anime_df['title'].values:
        return []
    
    # Lấy index của phim
    movie_idx = anime_df[anime_df['title'] == movie_title].index[0]
    
    # Nếu chưa có embeddings, tạo mới
    if movie_embeddings is None:
        print("Đang tạo embeddings cho tất cả phim...")
        movie_embeddings = []
        for idx, row in anime_df.iterrows():
            # Kết hợp các thông tin thành một chuỗi
            text = f"{row['title']} {row['genres']} {row['description']}"
            # Tokenize và tạo embedding
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            # Lấy embedding của [CLS] token
            embedding = outputs.last_hidden_state[0][0].numpy()
            movie_embeddings.append(embedding)
        movie_embeddings = np.array(movie_embeddings)
        
        # Lưu embeddings vào file
        print("Đang lưu embeddings vào file...")
        with open("bert_embeddings.pkl", "wb") as f:
            pickle.dump(movie_embeddings, f)
        print("Đã lưu embeddings thành công.")
    
    # Tính toán độ tương đồng cosine
    movie_embedding = movie_embeddings[movie_idx]
    similarities = cosine_similarity([movie_embedding], movie_embeddings)[0]
    
    # Lấy các phim tương tự nhất (bỏ qua chính nó)
    similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
    
    # Trả về danh sách các phim được gợi ý
    recommendations = []
    for idx in similar_indices:
        movie = anime_df.iloc[idx]
        recommendations.append({
            'title': movie['title'],
            'genres': movie['genres'],
            'description': movie['description'],
            'similarity_score': float(similarities[idx])
        })
    
    return recommendations

# Hàm get_simplified_collaborative_recommendations giữ nguyên như trước
def get_simplified_collaborative_recommendations(input_movie_title, df, top_n=10):
    """
    Gợi ý phim dựa trên một phương pháp "Collaborative Filtering" đơn giản hóa.
    Phương pháp này gợi ý các phim phổ biến khác (rating_score cao)
    trong cùng thể loại với phim đầu vào.
    
    LƯU Ý: Đây KHÔNG phải là Collaborative Filtering thực thụ vì thiếu dữ liệu
    tương tác của từng người dùng (user-item ratings).
    Một hệ thống CF đầy đủ cần ma trận user-item (user_id, movie_id, rating).
    """
    if df.empty or input_movie_title not in df['title'].values:
        print(f"Không tìm thấy phim '{input_movie_title}' trong database hoặc database rỗng cho simplified CF.")
        return []

    input_movie_details = df[df['title'] == input_movie_title].iloc[0]
    input_genres_str = input_movie_details['genres']
    input_genres = set(input_genres_str.split(', ')) if pd.notna(input_genres_str) and input_genres_str else set()
    
    if not input_genres:
        print(f"Phim '{input_movie_title}' không có thông tin thể loại để gợi ý.")
        return []

    recommended_movies = []
    for index, row in df.iterrows():
        if row['title'] == input_movie_title:
            continue
        
        current_movie_genres_str = row['genres']
        current_movie_genres = set(current_movie_genres_str.split(', ')) if pd.notna(current_movie_genres_str) and current_movie_genres_str else set()
        
        if input_genres.intersection(current_movie_genres):
            rating_score = row['rating_score']
            # rating_count đã được thêm vào df từ load_data_from_db
            rating_count = row['rating_count'] if 'rating_count' in df.columns and pd.notna(row['rating_count']) else 0
            recommended_movies.append((rating_score, rating_count, row['title']))
    
    recommended_movies.sort(key=lambda x: (x[0] is None, -float('inf') if x[0] is None else -x[0], x[1] is None, -float('inf') if x[1] is None else -x[1]))

    return [movie[2] for movie in recommended_movies[:top_n]]


if __name__ == "__main__":
    if bert_model is None or tokenizer is None:
        print("Thoát chương trình do BERT model/tokenizer không tải được.")
        exit()
        
    anime_df = load_data_from_db()

    if not anime_df.empty:
        target_anime_title = "One Piece" 

        if target_anime_title in anime_df['title'].values:
            print(f"\n--- Gợi ý Content-Based (BERT) cho '{target_anime_title}' ---")
            cb_bert_recommendations = get_content_based_recommendations_bert(target_anime_title, anime_df.copy())
            if cb_bert_recommendations:
                for i, rec_title in enumerate(cb_bert_recommendations):
                    print(f"{i+1}. {rec_title}")
            else:
                print("Không có gợi ý nào từ Content-Based (BERT).")

            print(f"\n--- Gợi ý Collaborative Filtering (Đơn giản hóa) cho '{target_anime_title}' ---")
            scf_recommendations = get_simplified_collaborative_recommendations(target_anime_title, anime_df.copy())
            if scf_recommendations:
                for i, rec_title in enumerate(scf_recommendations):
                    print(f"{i+1}. {rec_title}")
            else:
                print("Không có gợi ý nào từ Collaborative Filtering (Đơn giản hóa).")
        else:
            print(f"Phim '{target_anime_title}' không được tìm thấy trong cơ sở dữ liệu.")
    else:
        print("Không thể tải dữ liệu từ database để đưa ra gợi ý.") 