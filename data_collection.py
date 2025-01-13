import difflib
import os
import json
import string
import requests
import re
import bs4
from tqdm import tqdm

from scrt import newsapi_key

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"

def get_newsagencies(language="en", country="us"):
    response = requests.get(f"https://newsapi.org/v2/sources?apiKey={newsapi_key}&language={language}&country={country}")
    if response.status_code != 200:
        raise (f"Error: {response.status_code} \n", response.text)

    newsagencies = response.json()["sources"]
    return newsagencies

def create_data_management_infrastructure_for_newsagency(newsagency):
    os.makedirs(f"data/{newsagency['id']}", exist_ok=True)
    os.makedirs(f"data/{newsagency['id']}/documents", exist_ok=True)
    if not os.path.exists(f"data/{newsagency['id']}/skipped.json"):
        with open(f"data/{newsagency['id']}/skipped.json", "w") as f:
            json.dump([], f, indent=4)
    if not os.path.exists(f"data/{newsagency['id']}/metadata.json"):
        with open(f"data/{newsagency['id']}/metadata.json", "w") as f:
            json.dump([], f, indent=4)
    if not os.path.exists(f"data/{newsagency['id']}/newsagency_metadata.json"):
        with open(f"data/{newsagency['id']}/newsagency_metadata.json", "w") as f:
            json.dump(newsagency, f, indent=4)

def get_articles(newsagency_id, q="",page=1):
    response = requests.get(f"https://newsapi.org/v2/everything?q={q}&sortBy=popularity&apiKey={newsapi_key}&sources={newsagency_id}&page={page}")
    if response.status_code != 200:
        raise (f"Error: {response.status_code} \n", response.text)
    articles = response.json()["articles"]
    return articles

def get_full_article(article, min_match=100):
    response = requests.get(f"{article['url']}")

    full_website = bs4.BeautifulSoup(response.text, "html.parser").get_text()

    first_200_chars = article["content"][0:189]
    match = re.search(r'\[\+(\d+) chars\]', article["content"])
    article_length = 30000
    if match:
        article_length = int(match.group(1))+200
    sequence_matcher = difflib.SequenceMatcher(None, full_website, first_200_chars)
    match = sequence_matcher.find_longest_match(0, len(full_website), 0, len(first_200_chars))

    start_index = match.a if match.size > min_match else -1
    if start_index == -1:
        return article
    end_index = start_index + article_length
    full_article = full_website[start_index:end_index]
    full_article = article["title"] + "\n" + full_article

    article["full_article"] = full_article
    return article

def sanitize_filename(filename):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    sanitized = ''.join(c for c in filename if c in valid_chars)
    return sanitized

def save_article(newsagency, article):
    sanitized_title = sanitize_filename(article['title'])
    with open(f"data/{newsagency['id']}/documents/{sanitized_title}.txt", "w") as f:
        json.dump(article["full_article"], f, indent=4)

    del article["full_article"]
    del article["source"]

    article["source_id"] = newsagency["id"]
    article["source_name"] = newsagency["name"]
    article["source_description"] = newsagency["description"]
    article["source_url"] = newsagency["url"]
    article["source_category"] = newsagency["category"]
    article["source_language"] = newsagency["language"]
    article["source_country"] = newsagency["country"]
    article["txt_file"] = f"{sanitized_title}.txt"

    with open(f"data/{newsagency['id']}/metadata.json", "r") as f:
        metadata = json.load(f)
    metadata.append(article)
    with open(f"data/{newsagency['id']}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

def article_already_saved_or_skipped(newsagency, article):
    with open(f"data/{newsagency['id']}/skipped.json", "r") as f:
        skipped = json.load(f)
    if not os.path.exists(f"data/{newsagency['id']}/documents/{article['title']}.txt") and article["title"] not in skipped:
        return False
    else:
        return True

def skip_article(newsagency, article):
    with open(f"data/{newsagency['id']}/skipped.json", "r") as f:
        skipped = json.load(f)
    skipped.append(article["title"])
    with open(f"data/{newsagency['id']}/skipped.json", "w") as f:
        json.dump(skipped, f, indent=4)


def main():
    newsagencies = get_newsagencies()

    bar_format = f"{WHITE}⌛  Adding Text    {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=len(newsagencies), bar_format=bar_format, unit="newsagency") as pbar:
        for newsagency in newsagencies:
            create_data_management_infrastructure_for_newsagency(newsagency)
            articles = get_articles(newsagency["id"])

            for article in articles:
                if article_already_saved_or_skipped(newsagency, article):
                    skip_article(newsagency, article)
                    continue
                article = get_full_article(article)
                if "full_article" in article:
                    save_article(newsagency, article)
                else:
                    skip_article(newsagency, article)
            pbar.update(1)

if __name__ == "__main__":
    main()