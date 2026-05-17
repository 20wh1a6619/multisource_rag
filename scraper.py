import requests
from bs4 import BeautifulSoup

def scraper_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        clean_text = " ".join(text.split())
        return clean_text
    except:
        return "An error occurred while scraping the website."
