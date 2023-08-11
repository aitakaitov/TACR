from bs4 import BeautifulSoup

from utils import LibraryMethods


class CrawlerDrbnaAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "drbna"
        self.log_path = "log_drbna_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = ""
        self.max_scrolls = 42
        self.max_links = 100000
        self.is_ad = True

        self.parts = [
            ('https://www.drbna.cz/komercni-clanky.html', 64),
            ('https://plzenska.drbna.cz/komercni-clanky.html', 72),
            ('https://karlovarska.drbna.cz/komercni-clanky.html', 42),
            ('https://budejcka.drbna.cz/komercni-clanky.html', 272),
            ('https://jihlavska.drbna.cz/komercni-clanky.html', 71),
            ('https://brnenska.drbna.cz/komercni-clanky.html', 108),
            ('https://hanacka.drbna.cz/komercni-clanky.html', 114),
            ('https://hradecka.drbna.cz/komercni-clanky.html', 53),
            ('https://liberecka.drbna.cz/komercni-clanky.html', 136),
            ('https://prazska.drbna.cz/komercni-clanky.html', 72)
        ]

    def collect_links(self, driver):
        links = []
        titles = []

        for site, page_max in self.parts[:1]:
            for i in range(1, page_max + 1):
                url = f'{site}?strana={i}'

                html = LibraryMethods.download_page_html(driver, url, self.max_scrolls)
                soup = BeautifulSoup(html)

                self.get_article_urls(soup, links, titles)

        return links

    def get_article_urls(self, soup, links, titles):
        div_tags = soup.find_all("div", {'class': 'row loop loop--inline loopArticle mx-0'})
        for tag in div_tags:
            h_tag = tag.find('h2', {'class': 'mb-0 loop__heading'})
            if h_tag is None:
                continue

            a_tag = h_tag.find('a')
            if a_tag is None:
                continue

            title = a_tag.get('title')
            tag_url = a_tag.get("href")

            if tag_url not in links and title not in titles:
                links.append(tag_url)
                titles.append(title)
            else:
                pass

        return links

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("p", {"class": "alert alert-light bg-light"})
        if tag is not None:
            tag.extract()

        tag = soup.find("p", {"class": "detail__date mb-0 align-self-lg-center pr-3"})
        if tag is not None:
            tag.extract()
