import urllib.parse
import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


class CrawlerTiscaliAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "tiscali"
        self.log_path = "log_tiscali_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://zpravy.tiscali.cz/komercni-sdeleni"
        self.max_scrolls = 500
        self.max_links = 10000
        self.is_ad = True

    def collect_links(self, driver):
        driver.get(self.starting_page)
        consent = driver.find_element(By.XPATH, "//button[@aria-label='Souhlas']")
        consent.click()
        time.sleep(2)

        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        element = driver.find_element(By.XPATH, '(//button)[5]')
        element.click()

        for i in range(self.max_scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        html = driver.page_source
        soup = BeautifulSoup(html)

        links = self.get_article_urls(soup, self.starting_page)
        print(len(links))

        return links


    def get_article_urls(self, soup, url):
        links = []

        article_tags = soup.find_all("h3", {'class': 'media-title h2'})
        for tag in article_tags:
            a_tag = tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("span", {"class": "badge-list"})
        if tag is not None:
            tag.extract()

        tag = soup.find("p", {"class": "article-meta"})
        if tag is not None:
            tag.extract()

        tags = soup.find_all('a', {'data-ga-action': 'article-tag'})
        for tag in tags:
            tag.extract()

        tag = soup.find('div', {'class': 'article-sources row no-gutters-xs mb-1'})
        if tag is not None:
            tag.extract()

