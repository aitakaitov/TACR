from bs4 import BeautifulSoup
from selenium.common import WebDriverException

from utils import LibraryMethods


class CrawlerEmiminoArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "emimino"
        self.log_path = "log_emimino_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.emimino.cz/"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = False

        self.rubriky = [
            ('clanky/pr', 1),
            ('denicky', 2),
            ('clanky', 2),
        ]

    def collect_links(self, driver):
        print('collecting links')
        links = []

        for rubrika, pages in self.rubriky:
            for i in range(1, pages + 1):
                url = f'{self.starting_page}{rubrika}/strankovani/{i}/'
                try:
                    html = LibraryMethods.download_page_html(driver, url, self.max_scrolls)
                except WebDriverException:
                    break
                soup = BeautifulSoup(html)

                links.extend(self.get_article_urls(soup, None))

                print(f'Collected {len(links)} links')

        return links

    def get_article_urls(self, soup, url):
        links = []

        articles = soup.find_all("article", {'role': 'article'})
        for tag in articles:
            meta_tag = tag.find('ul', {'class': 'about-info'})
            if meta_tag is not None:
                if 'komerční sdělení' in meta_tag.get_text():
                    continue

            if 'article--premium' in tag['class'] or 'article--advert' in tag['class']:
                continue

            a_tag = tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(tag_url)

        return links

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("ul", {"class": "about-info"})
        if tag is not None:
            tag.extract()
