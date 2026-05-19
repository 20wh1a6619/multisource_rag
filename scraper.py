import requests
from bs4 import BeautifulSoup
import trafilatura

def scraper_website(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return "Error: Could not download webpage content."
        text = trafilatura.extract(downloaded)
        if not text:
            return "Error: No readable content found."
        return text
    except Exception as e:
        return f"Error: {str(e)}"

