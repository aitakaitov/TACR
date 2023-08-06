import time

from bs4 import BeautifulSoup
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By

from utils import LibraryMethods


class CrawlerNovinkyAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "novinky"
        self.log_path = "log_novinky_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.novinky.cz/komercni-clanky"
        self.max_scrolls = 42
        self.max_links = 100  #49 # for some reason it is impossible to crawl more links, dont know why
        self.is_ad = True

        self.list_pages = 50

    def collect_links(self, driver):
        for i in range(self.list_pages):
            element = driver.find_element(By.LINK_TEXT, 'Zobrazit další')
            element.click()
            time.sleep(3)

        try:
            html = LibraryMethods.download_page_html(driver, self.starting_page, self.max_scrolls)
        except WebDriverException:
            exit(-1)

        soup = BeautifulSoup(html)
        return self.get_article_urls(soup, self.starting_page)

    def get_article_urls(self, soup, url):
        links = []
        tags = soup.find_all('article', {'class': 'q_gO q_hn n_gO document-item--default'})
        for tag in tags:
            a_tag = tag.find('a', {'class': 'c_an q_g2'})

            tag_url = a_tag.get("href")
            if tag_url not in links:
                if tag_url not in links:
                    links.append(tag_url)

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("div", {"data-dot": "ogm-article-header"}).get_text()
        header = soup.find("p", {"data-dot": "ogm-article-perex"}).get_text()
        article_tag = soup.find("article", {"aria-labelledby": "accessibility-article"})
        tags = article_tag.find_all()

        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = article_tag.contents
        content_string = ""
        for i in range(len(content) - 1):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return title + "\n" + header + "\n" + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"data-qipojriimlnpiljnkqo": "atm-label"})
        if tag is not None:
            tag.extract()

        tag = soup.find('a', {'class': 'c_an g_eV'})
        if tag is not None:
            tag.extract()

        tag = soup.find('div', {'class': 'data-qipojriimlnpiljnkqo'})
        if tag is not None:
            tag.extract()

    def get_next_page(self, soup, url):
        expand = soup.find('div', {'class': 'atm-expand-button g_g7 g_hd'})
        if expand is None:
            return None

        url = expand.find('a')
        if url is None:
            return None

        return url.get('href')

    def check_soup(self, soup):
        return True
