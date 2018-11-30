from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import urllib.request
import time

def get_images_for_category(category, num_scrolls):
    browser = webdriver.Chrome()
    browser.get("https://unsplash.com/search/photos/" + category)
    time.sleep(1)
    elem = browser.find_element_by_tag_name("body")
    no_of_pagedowns = num_scrolls

    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns -= 1

    html_doc = browser.page_source
    browser.quit()

    soup = BeautifulSoup(html_doc, 'html.parser')
    all_pics = soup.find_all("img")

    sources = []
    for pic in all_pics:
        if "https://images.unsplash.com/photo-" in pic["src"]:
            sources.append(pic["src"])
    return sources

def scrape_images(categories):
    image_dict = {}
    for category in categories:
        image_dict[category] = get_images_for_category(category, 10)
    print(image_dict)

scrape_images(["mountain", "ocean", "island", "city"])