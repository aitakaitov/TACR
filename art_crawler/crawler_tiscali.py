import time
import urllib.parse

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By


class CrawlerTiscaliArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "tiscali"
        self.log_path = "log_tiscali_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://zpravy.tiscali.cz/"
        self.max_scrolls = 3000
        self.max_links = 10000
        self.is_ad = False

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

        article_tags = soup.find_all("div", {'class': 'media-object media media-row media-article-standard col-12'})
        for tag in article_tags:
            date_tag = tag.find('a', {'data-ga-action': 'kategorie-clanku'})
            if date_tag is not None:
                if date_tag['title'] == 'Komerční sdělení':
                    continue

            a_tag = tag.find('a', {'data-ga-action': 'article-detail'})
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
