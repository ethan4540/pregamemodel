import os.path
import bs4

import requests as r

save_path = '.'


def open_page(url):
    site = r.get(url)
    page_html = site.content
    page = bs4.BeautifulSoup(page_html, "html.parser")
    return page

def open_xml(url):
    site = r.get(url)
    page_html = site.text
    page = bs4.BeautifulSoup(page_html, "lxml")
    return page


def open_file(file_name):
    file = open(file_name, "w", encoding="utf-8")
    return file