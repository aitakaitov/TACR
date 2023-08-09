from bs4 import BeautifulSoup
from selenium.common import WebDriverException

from utils import LibraryMethods


class CrawlerForum24Art:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "forum24"
        self.log_path = "log_forum24_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.forum24.cz/rubrika/"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.rubriky = [
            ('rozhovory', 70),
            ('svobodne-forum', 200),
            ('kultura', 45),
            ('zpravy', 400)
        ]

    def collect_links(self, driver):
        print('collecting links')
        links = []

        for rubrika, pages in self.rubriky:
            for i in range(1, pages + 1):
                url = f'{self.starting_page}{rubrika}?stranka={i}'
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
        div_tags = soup.find_all("article")
        for tag in div_tags:
            a_tag = tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(tag_url)

        return links

    def check_soup(self, soup):
        meta = soup.find('span', {'class': 'text-brand-blue/70'})
        if meta is not None and 'Komerční sdělení' in meta.get_text():
            return False

        return True

    def remove_article_heading(self, soup):
        tag = soup.find("span", {"class": "text-brand-blue/70"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "space-y-2"})
        if tag is not None:
            tag.extract()

