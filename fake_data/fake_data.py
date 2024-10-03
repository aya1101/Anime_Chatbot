from faker import Faker
import random

fake = Faker('vi_VN')
genre_option = []
status_options = ["Hoàn thành", "Đang tiến hành"]
def generate_fake_anime(num):
    movie_list = []
    for i in range(1, num ) :
        movie = {
            "anime_id": i,
            "title" :fake.catch_phrase ,
            "genre" : random.sample(genre_option),
            "status": random.choice(status_options),
            "eposides": random.choice(range(0,300)),
            "realese_year": random.randint(1995, 2024),
            "description": fake.text(max_nb_chars=200)
        }
        movie_list.append(movie)
    return movie_list


fake_anime = generate_fake_anime(3)

for movie in fake_anime:
    print(movie)
