from platform import release

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


def check_page_not_found(soup):
    return soup.find({"class": "ah_404"}) is not None

def fetch_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        page_soup = BeautifulSoup(response.content, 'html.parser')

        if check_page_not_found(page_soup):
            print('Fail - Page Not Found 404')
            return None

        return page_soup
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def save_soup(soup, file_name):
    with open(file_name + '.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

def fetch_movie_title(soup):
    title = soup.find('h1', {"class": "heading_movie"}).text if soup.find('h1', {"class": "heading_movie"}) else 'N/A'
    return title

def fetch_movie_genre(soup):
    genre = soup.find('div', {"class": "list_cate"}).text if soup.find('div', {"class": "list_cate"}) else 'N/A'
    genre_list = genre.splitlines()
    clean_genre_list = [g.strip() for g in genre_list if len(g.strip()) > 0][1:]
    return clean_genre_list

def fetch_movie_rating(soup):
    rating = soup.find('div', {"class": "score"}).text if soup.find('div', {"class": "score"}) else 'N/A'
    if rating != 'N/A':
        rating_parts = rating.split()
        if len(rating_parts) >= 4:
            return [rating_parts[1], rating_parts[3]]
        else:
            return 'Invalid rating format'
    else:
        return rating

def fetch_movie_status(soup):
    status = soup.find('div', {"class": "status"})
    if status:
        status = status.find_all('div')[-1].text.strip()
    else:
        status = 'Undefined'
    return status

def fetch_movie_release_year(soup):
    release_year = soup.find('div', {"class": "update_time"})
    if release_year:
        release_year = release_year.find_all('div')[-1].text.strip()
    else:
        release_year = 'N/A'
    return release_year

def fetch_movie_information(soup):
    title = fetch_movie_title(soup)
    genre = fetch_movie_genre(soup)
    rating = fetch_movie_rating(soup)
    status = fetch_movie_status(soup)
    release_year = fetch_movie_release_year(soup)
    movie = {
        'title': title,
        'genre': genre,
        'rating': rating,
        'status': status,
        'release year': release_year
    }
    return movie


def fetch_pages(start_page, end_page):
    current_page = start_page
    film_list = []
    while current_page <= end_page:
        url = f"https://animehay.cam/the-loai/anime-{current_page}.html"
        print(f"Processing on page: ", url)

        soup = fetch_soup(url)
        if soup is None:
            break

        film_list.append(soup)
        current_page += 1
        print('\t----------\t----------\t---------\t----------\t----------\t---------')

    return film_list

def get_all_movies_links(soup):
    links = soup.find_all('a', href = True)
    specific_links = [link['href'] for link in links if "thong-tin-phim" in link['href']]
    return specific_links

def fetch_all_movies(list_links, save=False):
    all_movies = {}
    for link in list_links:
        try:
            print(f"Processing link: {link}")
            movie_soup = fetch_soup(link)
            if not movie_soup:
                print(f"Failed to fetch data from {link}")
                continue

            information = fetch_movie_information(movie_soup)

            if save:
                save_soup(movie_soup, information['title'])

            all_movies[information['title']] = information

            print(f"Successfully processed: {information['title']}")
            print('----------\t---------\t----------')

        except Exception as ex:
            print(f"Error while processing link {link}: {ex}")

    return all_movies

def save_json(data, filename):
    with open(filename + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def json_to_csv(json_filename, csv_filename):
    with open(json_filename + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')

    df.to_csv(csv_filename + '.csv', encoding='utf-8', index=False)

all_soups = fetch_pages(1, 150)

all_links = []
for soup in all_soups:
    specific_links_soup = get_all_movies_links(soup)
    all_links.extend(specific_links_soup)

all_movies = fetch_all_movies(all_links)
save_json(all_movies, 'data_movies')
json_to_csv('data_movies', 'movies_data')
