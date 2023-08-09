from bs4 import BeautifulSoup
from selenium.common import WebDriverException

from utils import LibraryMethods


class CrawlerAhaonlineArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "ahaonline"
        self.log_path = "log_ahaonline_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.ahaonline.cz/kategorie/"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.rubriky = [
            ('2524/malery', 26),
            ('2511/zhave-drby', 30),
            ('2513/musite-vedet', 30)
        ]

    def collect_links(self, driver):
        print('collecting links')
        links = []

        for rubrika, pages in self.rubriky:
            for i in range(1, pages + 1):
                url = f'{self.starting_page}{rubrika}?page={i}'
                try:
                    html = LibraryMethods.download_page_html(driver, url, self.max_scrolls)
                except WebDriverException:
                    break
                soup = BeautifulSoup(html)

                links.extend(self.get_article_urls(soup, None))

                if len(links) > self.max_links:
                    break

                print(f'Collected {len(links)} links')

        return links

    def get_article_urls(self, soup, url):
        links = []
        div_tags = soup.find_all("article", {'class': 'inner_8 behavSource'})
        for tag in div_tags:
            a_tag = tag.find('a', {'class': 'readMore floatRight'})
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")

            if 'https://www.ahaonline.cz/' not in tag_url:
                continue

            if tag_url not in links:
                links.append(tag_url)

        return links

    def check_soup(self, soup):
        meta = soup.find('div', {'class': 'articleMeta'})
        if meta is not None and 'Prezentace klienta' in meta.get_text():
            return False

        return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "articleMeta"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "article-keywords"})
        if tag is not None:
            tag.extract()
