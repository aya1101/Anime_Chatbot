import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import io
import json

# Cấu hình encoding cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def load_data():
    """Tải dữ liệu từ file JSON."""
    try:
        # Sử dụng đường dẫn tương đối
        json_path = os.path.join('data', 'data_movies.json')
        
        print(f"Đang tìm file JSON tại: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"Không tìm thấy file tại: {json_path}")
            return pd.DataFrame()
            
        # Đọc file JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Chuyển đổi dữ liệu thành DataFrame
        movies_list = []
        for movie_id, movie_data in data.items():
            movie_data['movie_id'] = movie_id
            # Chuyển đổi genre từ list thành string
            movie_data['genres'] = ', '.join(movie_data['genre'])
            # Tách rating thành score và count
            movie_data['rating_score'] = float(movie_data['rating'][0])
            movie_data['rating_count'] = int(movie_data['rating'][1])
            # Xóa các trường không cần thiết
            del movie_data['genre']
            del movie_data['rating']
            movies_list.append(movie_data)
        
        df = pd.DataFrame(movies_list)
        print(f"Đã tải {len(df)} phim từ file JSON.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {str(e)}")
        return pd.DataFrame()

def get_content_based_recommendations_tfidf(movie_title, df, top_n=5):
    """Gợi ý phim dựa trên nội dung sử dụng TF-IDF."""
    if movie_title not in df['title'].values:
        return []
    
    # Tạo corpus từ title, genres và description
    df['content'] = df['title'] + ' ' + df['genres'].fillna('') + ' ' + df['description'].fillna('')
    
    # Khởi tạo TF-IDF vectorizer
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

def get_genre_based_recommendations(genre, df, top_n=5):
    """Gợi ý phim dựa trên thể loại."""
    if df.empty:
        return []
    
    # Tìm các phim có chứa thể loại được chọn
    genre_movies = df[df['genres'].str.contains(genre, case=False, na=False)]
    
    if genre_movies.empty:
        return []
    
    # Sắp xếp theo rating_score (nếu có) hoặc rating_count
    if 'rating_score' in genre_movies.columns:
        genre_movies = genre_movies.sort_values('rating_score', ascending=False)
    
    # Lấy top N phim
    recommendations = []
    for _, movie in genre_movies.head(top_n).iterrows():
        recommendations.append({
            'title': movie['title'],
            'genres': movie['genres'],
            'description': movie['description'],
            'rating_score': movie.get('rating_score', 'N/A')
        })
    
    return recommendations

def main():
    # Tải dữ liệu
    df = load_data()
    if df.empty:
        print("Không thể tải dữ liệu. Vui lòng kiểm tra file JSON.")
        return

    while True:
        print("\n=== HỆ THỐNG GỢI Ý PHIM ===")
        print("1. Gợi ý phim dựa trên phim yêu thích")
        print("2. Gợi ý phim dựa trên thể loại")
        print("3. Thoát")
        
        choice = input("\nChọn chức năng (1-3): ")
        
        if choice == '1':
            movie_title = input("\nNhập tên phim yêu thích: ")
            print(f"\nĐang tìm phim tương tự với '{movie_title}'...")
            recommendations = get_content_based_recommendations_tfidf(movie_title, df)
            
            if recommendations:
                print("\nCác phim được gợi ý:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['title']}")
                    print(f"   Thể loại: {rec['genres']}")
                    print(f"   Mô tả: {rec['description'][:200]}...")
                    print(f"   Độ tương đồng: {rec['similarity_score']:.2f}")
            else:
                print(f"Không tìm thấy phim '{movie_title}' hoặc không có gợi ý phù hợp.")
        
        elif choice == '2':
            print("\nCác thể loại phim phổ biến:")
            genres = set()
            for genre_list in df['genres'].dropna():
                genres.update(genre_list.split(', '))
            print(', '.join(sorted(genres)))
            
            genre = input("\nNhập thể loại phim yêu thích: ")
            print(f"\nĐang tìm phim thể loại '{genre}'...")
            recommendations = get_genre_based_recommendations(genre, df)
            
            if recommendations:
                print("\nCác phim được gợi ý:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['title']}")
                    print(f"   Thể loại: {rec['genres']}")
                    print(f"   Mô tả: {rec['description'][:200]}...")
                    if 'rating_score' in rec:
                        print(f"   Đánh giá: {rec['rating_score']}")
            else:
                print(f"Không tìm thấy phim nào thuộc thể loại '{genre}'.")
        
        elif choice == '3':
            print("\nCảm ơn bạn đã sử dụng hệ thống gợi ý phim!")
            break
        
        else:
            print("\nLựa chọn không hợp lệ. Vui lòng chọn lại.")

if __name__ == "__main__":
    main() 