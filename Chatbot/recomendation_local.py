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
        json_path = os.path.join('data', 'data_movies.json')
        
        print(f"Đang tìm file JSON tại: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"Không tìm thấy file tại: {json_path}")
            return pd.DataFrame()
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        
        movies_list = []
        for movie_id, movie_data in data.items():
            movie_data['movie_id'] = movie_id
            # list -> string
            movie_data['genres'] = ', '.join(movie_data['genre'])
        
            movie_data['rating_score'] = float(movie_data['rating'][0])
            movie_data['rating_count'] = int(movie_data['rating'][1])
            
            del movie_data['genre']
            del movie_data['rating']
            movies_list.append(movie_data)
        
        df = pd.DataFrame(movies_list)
        print(f"Đã tải {len(df)} phim từ file JSON.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {str(e)}")
        return pd.DataFrame()

def load_vietnamese_stopwords(filepath="data/stopwords-vi.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]
    return stopwords

def get_content_based_recommendations_tfidf(movie_title, df, top_n):
    """Gợi ý phim dựa trên nội dung sử dụng TF-IDF."""
    if movie_title not in df['title'].values:
        return []
    
    # Tạo corpus từ title, genres và description
    df['content'] = df['title'] + ' ' + df['genres'].fillna('') + ' ' + df['description'].fillna('')
    vietnamese_stopwords = load_vietnamese_stopwords()
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=vietnamese_stopwords,
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])
    
    
    movie_idx = df[df['title'] == movie_title].index[0]
    
    # độ tương đồng cosine
    movie_vector = tfidf_matrix[movie_idx]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Lấ top n
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    
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

def get_genre_based_recommendations(genre, df, top_n):
    """Gợi ý phim dựa trên thể loại."""
    if df.empty:
        return []
    
    # Filtering 
    genre_movies = df[df['genres'].str.contains(genre, case=False, na=False)]
    
    if genre_movies.empty:
        return []
    
    # Sắp xếp theo rating_score/rating_count
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
            try:
                top_n = int(input("Nhập số lượng phim muốn đề xuất (mặc định 5): ") or 5)
            except ValueError:
                top_n = 5
            recommendations = get_content_based_recommendations_tfidf(movie_title, df, top_n=top_n)
            
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
            try:
                top_n = int(input("Nhập số lượng phim muốn đề xuất (mặc định 10): ") or 10)
            except ValueError:
                top_n = 10
            recommendations = get_genre_based_recommendations(genre, df, top_n=top_n)
            
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
            print("\nBýe!")
            break
        
        else:
            print("\nLựa chọn không hợp lệ.")

if __name__ == "__main__":
    main() 