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
        self.max_links = 30
        self.is_ad = False

        self.page = 1
        self.page_max = 1050

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'row loop loop--inline loopArticle mx-0'})
        for tag in div_tags:
            h_tag = tag.find('h2', {'class': 'mb-0 loop__heading'})
            if h_tag is None:
                continue

            a_tag = h_tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(tag_url)

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
