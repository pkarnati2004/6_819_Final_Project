from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib
import time
import os

# given category and list of image urls within category
# saves images at corresponding path
def download_images(category, sources):
    if not os.path.isdir("./images"):
        os.mkdir("./images")
    if os.path.isdir("./images/" + category):
        print("./images/" + category + " already exists. Will skip category.")
        return
    os.mkdir("./images/" + category)
    counter = 1
    for url in sources:
        path = "./images/" + category + "/" + "0" * (8 - len(str(counter))) + str(counter) + ".jpg"
        f = open(path, "wb")
        with urllib.request.urlopen(url) as opened:
            f.write(opened.read())
        f.close()
        print("saved " + path)
        counter += 1


# removes parameters from image urls, sets image download width
def adjust_size(url, width):
    no_params = url.split("?")
    return no_params[0] + "?w=" + str(width)


# returns a list of adjusted-size urls for specific category
# num_scrolls is the number of physical scrolls on the page
# (correlates to the number of image urls you will scrape from the webpage)
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

    soup = BeautifulSoup(html_doc, "html.parser")
    all_pics = soup.find_all("img")

    sources = []
    for pic in all_pics:
        if "https://images.unsplash.com/photo-" in pic["src"]:
            sources.append(adjust_size(pic["src"], 512))
    return sources


# wrapper function: scrapes for each categories
# saves images to './images/category_name/00000num.jpg'
def scrape_images(categories):
    category_to_urls = {}
    for category in categories:
        category_to_urls[category] = get_images_for_category(category, 10)
        download_images(category, category_to_urls[category])
    return category_to_urls


scrape_images(["mountain", "ocean", "island", "city"])
