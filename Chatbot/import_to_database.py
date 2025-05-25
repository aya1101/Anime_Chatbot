import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import mysql.connector
from mysql.connector import errorcode

# Cấu hình kết nối MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Thay bằng username MySQL của bạn
    'password': '',  # Thay bằng password MySQL của bạn
    'database': 'movies_db'  # Tên database bạn muốn tạo hoặc sử dụng
}
JSON_FILE_PATH = 'data/data_movies.json'
TABLE_NAME = 'movies'

def create_database_and_table():
    """Tạo database và bảng nếu chúng chưa tồn tại."""
    try:
        # Kết nối đến MySQL server (không chỉ định database ban đầu để có thể tạo database)
        temp_config = DB_CONFIG.copy()
        db_name = temp_config.pop('database') #Lấy tên database ra khỏi config tạm thời

        cnx = mysql.connector.connect(**temp_config)
        cursor = cnx.cursor()

        try:
            cursor.execute(f"CREATE DATABASE {db_name} DEFAULT CHARACTER SET utf8mb4")
            print(f"Database '{db_name}' đã được tạo.")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_DB_CREATE_EXISTS:
                print(f"Database '{db_name}' đã tồn tại.")
            else:
                print(err)
                exit(1)

        # Kết nối lại với database đã được chỉ định hoặc vừa tạo
        cnx.database = db_name

        # Tạo bảng movies
        table_description = (
            f"CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` ("
            "  `id` INT AUTO_INCREMENT PRIMARY KEY,"
            "  `title` VARCHAR(255) NOT NULL UNIQUE,"
            "  `genre` TEXT,"
            "  `rating_score` FLOAT,"
            "  `rating_count` INT,"
            "  `status` VARCHAR(50),"
            "  `episodes` INT,"
            "  `release_year` INT,"
            "  `description` TEXT"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        try:
            print(f"Đang tạo bảng `{TABLE_NAME}`...")
            cursor.execute(table_description)
            print(f"Bảng `{TABLE_NAME}` đã được tạo hoặc đã tồn tại.")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print(f"Bảng `{TABLE_NAME}` đã tồn tại.")
            else:
                print(err.msg)
        finally:
            cursor.close()
            cnx.close()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Lỗi: Sai username hoặc password MySQL.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Lỗi: Database '{DB_CONFIG['database']}' không tồn tại và không thể tạo.")
        else:
            print(err)
        exit(1)

def insert_data_to_mysql():
    """Đọc dữ liệu từ JSON và chèn vào MySQL."""
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        print(f"Đã kết nối tới database '{DB_CONFIG['database']}'.")

        # Đọc dữ liệu từ file JSON
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        insert_query = (
            f"INSERT INTO `{TABLE_NAME}` "
            "(title, genre, rating_score, rating_count, status, episodes, release_year, description) "
            "VALUES (%(title)s, %(genre)s, %(rating_score)s, %(rating_count)s, %(status)s, %(episodes)s, %(release_year)s, %(description)s)"
        )

        movies_added = 0
        movies_skipped = 0

        for movie_title, details in data.items():
            try:
                # Làm sạch và chuẩn hóa dữ liệu trước khi chèn
                cleaned_title = details.get('title')
                if cleaned_title:
                    cleaned_title = cleaned_title.strip()

                cleaned_genres = []
                if isinstance(details.get('genre'), list):
                    for g in details.get('genre'):
                        if isinstance(g, str):
                            cleaned_genres.append(g.strip().title()) # Bỏ khoảng trắng, viết hoa chữ cái đầu
                genre_str = ', '.join(cleaned_genres) if cleaned_genres else None

                rating_list = details.get('rating', [None, None])
                rating_score_val = None
                if rating_list and rating_list[0] is not None:
                    try:
                        rating_score_val = float(rating_list[0])
                    except (ValueError, TypeError):
                        pass # Giữ None nếu không chuyển đổi được

                rating_count_val = None
                if rating_list and len(rating_list) > 1 and rating_list[1] is not None:
                    try:
                        rating_count_val = int(rating_list[1])
                    except (ValueError, TypeError):
                        pass # Giữ None nếu không chuyển đổi được

                cleaned_status = details.get('status')
                if cleaned_status:
                    cleaned_status = cleaned_status.strip().title() # Bỏ khoảng trắng, viết hoa chữ cái đầu

                episodes_val = None
                if details.get('episodes') is not None:
                    try:
                        episodes_val = int(details.get('episodes'))
                    except (ValueError, TypeError):
                        pass # Giữ None nếu không chuyển đổi được
                
                release_year_val = None
                raw_release_year = details.get('release year')
                if raw_release_year and str(raw_release_year).isdigit():
                    try:
                        release_year_val = int(raw_release_year)
                    except (ValueError, TypeError):
                        pass # Giữ None nếu không chuyển đổi được

                cleaned_description = details.get('description')
                if cleaned_description:
                    cleaned_description = cleaned_description.strip()
                    if cleaned_description.upper() == 'N/A':
                        cleaned_description = None

                movie_data = {
                    'title': cleaned_title,
                    'genre': genre_str,
                    'rating_score': rating_score_val,
                    'rating_count': rating_count_val,
                    'status': cleaned_status,
                    'episodes': episodes_val,
                    'release_year': release_year_val,
                    'description': cleaned_description
                }

                # Bỏ qua việc chèn nếu title là None hoặc rỗng sau khi làm sạch
                if not cleaned_title:
                    print(f"Phim có tiêu đề không hợp lệ hoặc rỗng đã bị bỏ qua. Dữ liệu gốc: {details}")
                    movies_skipped += 1
                    continue

                cursor.execute(insert_query, movie_data)
                movies_added += 1
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DUP_ENTRY:
                    print(f"Phim '{movie_title}' đã tồn tại, bỏ qua.")
                    movies_skipped += 1
                else:
                    print(f"Lỗi khi chèn phim '{movie_title}': {err}")
                    movies_skipped += 1
            except ValueError as ve:
                print(f"Lỗi giá trị khi xử lý phim '{movie_title}': {ve}. Dữ liệu: {details}")
                movies_skipped +=1


        cnx.commit()
        print(f"Hoàn tất! Đã thêm {movies_added} phim. Bỏ qua {movies_skipped} phim.")

    except mysql.connector.Error as err:
        print(f"Lỗi MySQL: {err}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{JSON_FILE_PATH}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()
            print("Đã đóng kết nối MySQL.")

if __name__ == '__main__':
    create_database_and_table()
    insert_data_to_mysql() 