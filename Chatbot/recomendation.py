import sys
import io
import pickle

# Cấu hình stdout để sử dụng UTF-8, giải quyết vấn đề UnicodeEncodeError trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Cấu hình kết nối MySQL (Giống với import_to_database.py)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Thay bằng username MySQL của bạn
    'password': '123456',  # Thay bằng password MySQL của bạn
    'database': 'anime_db' # Tên database bạn đã tạo
}
TABLE_NAME = 'anime'

# Biến toàn cục để lưu trữ TF-IDF matrix
tfidf_matrix = None
tfidf_vectorizer = None

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

def get_content_based_recommendations_tfidf(movie_title, df, top_n=5):
    """
    Gợi ý phim dựa trên nội dung sử dụng TF-IDF.
    
    Args:
        movie_title (str): Tên phim cần gợi ý
        df (DataFrame): DataFrame chứa dữ liệu phim
        top_n (int): Số lượng phim gợi ý
        
    Returns:
        list: Danh sách các phim được gợi ý
    """
    global tfidf_matrix, tfidf_vectorizer
    
    if movie_title not in df['title'].values:
        return []
    
    # Tạo corpus từ title, genres và description
    df['content'] = df['title'] + ' ' + df['genres'].fillna('') + ' ' + df['description'].fillna('')
    
    # Khởi tạo TF-IDF vectorizer nếu chưa có
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    
    # Lấy index của phim
    movie_idx = df[df['title'] == movie_title].index[0]
    
    # Tính toán độ tương đồng cosine
    movie_vector = tfidf_matrix[movie_idx]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Lấy các phim tương tự nhất (bỏ qua chính nó)
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    
    # Trả về danh sách các phim được gợi ý
    recommendations = []
    for idx in similar_indices:
        movie = df.iloc[idx]
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
    anime_df = load_data_from_db()

    if not anime_df.empty:
        target_anime_title = "One Piece" 

        if target_anime_title in anime_df['title'].values:
            print(f"\n--- Gợi ý Content-Based (TF-IDF) cho '{target_anime_title}' ---")
            cb_tfidf_recommendations = get_content_based_recommendations_tfidf(target_anime_title, anime_df.copy())
            if cb_tfidf_recommendations:
                for i, rec in enumerate(cb_tfidf_recommendations):
                    print(f"{i+1}. {rec['title']} (Độ tương đồng: {rec['similarity_score']:.2f})")
            else:
                print("Không có gợi ý nào từ Content-Based (TF-IDF).")

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