import os.path
import bs4

import requests as r

save_path = '.'

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

def page(url):
    site = r.get(url, headers=headers)
    page_html = site.content
    page = bs4.BeautifulSoup(page_html, "html.parser")
    return page


def xml(url):
    site = r.get(url)
    page_html = site.text
    page = bs4.BeautifulSoup(page_html, "lxml")
    return page


def file(file_name):
    file = open(file_name, "w", encoding="utf-8")
    return file
