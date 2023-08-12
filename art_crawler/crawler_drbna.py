from bs4 import BeautifulSoup

from utils import LibraryMethods


class CrawlerDrbnaArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "drbna"
        self.log_path = "log_drbna_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.drbna.cz/aktualne.html"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.parts = [
            ('https://plzenska.drbna.cz/aktualne.html', 120),
            ('https://karlovarska.drbna.cz/aktualne.html', 120),
            ('https://budejcka.drbna.cz/aktualne.html', 120),
            ('https://jihlavska.drbna.cz/aktualne.html', 120),
            ('https://brnenska.drbna.cz/aktualne.html', 120),
            ('https://hanacka.drbna.cz/aktualne.html', 120),
            ('https://hradecka.drbna.cz/aktualne.html', 120),
            ('https://liberecka.drbna.cz/aktualne.html', 120),
            ('https://prazska.drbna.cz/aktualne.html', 120)
        ]

    def collect_links(self, driver):
        links = []
        titles = []

        for site, page_max in self.parts:
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

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}?strana={self.page}'


    def check_soup(self, soup):
        pr_warning = soup.find('p', {'class': 'alert alert-light bg-light'})
        if pr_warning is not None:
            if 'PR ' in pr_warning.get_text():
                return False

        pr_tag = soup.find("p", {"class": "detail__date mb-0 align-self-lg-center pr-3"})
        if pr_tag is not None:
            if 'PR ' in pr_tag.get_text():
                return False

        return True

    def remove_article_heading(self, soup):
        tag = soup.find("p", {"class": "alert alert-light bg-light"})
        if tag is not None:
            tag.extract()

        tag = soup.find("p", {"class": "detail__date mb-0 align-self-lg-center pr-3"})
        if tag is not None:
            tag.extract()
