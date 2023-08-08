from selenium.common.exceptions import WebDriverException
from utils.library_methods import LibraryMethods
from bs4 import BeautifulSoup


class CrawlerIdnesAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "idnes"
        self.log_path = "log_idnes_ad.log"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://sdeleni.idnes.cz/"
        self.max_scrolls = 10
        self.is_ad = True

        self.rubriky = [
            ('ona', 3),
            ('zpravy', 21),
            ('technet', 2),
            ('ekonomika', 4),
            ('bydleni', 4),
            ('cestovani', 2),
            ('praha', 4),
            ('brno', 2),
            ('hradec', 3),
            ('jihlava', 2),
            ('ostrava', 3),
            ('usti', 3),
        ]

    def collect_links(self, driver):
        print("Collecting links")
        links = []

        for rubrika, pages in self.rubriky:
            for i in range(1, pages + 1):
                url = f'{self.starting_page}/{rubrika}/{i}'
                try:
                    html = LibraryMethods.download_page_html(driver, url, self.max_scrolls)
                except WebDriverException:
                    break
                soup = BeautifulSoup(html)

                div_tags = soup.find_all("div", {"class": "art"}) + soup.find_all("div", {"class": "art opener"})
                for tag in div_tags:
                    a_tag = tag.find("a")

                    if a_tag is None:
                        continue

                    tag_url = a_tag.get("href")
                    if tag_url not in links:
                        if "idnes.cz" in LibraryMethods.strip_url(tag_url):
                            links.append(tag_url)

                print(f'Collected {len(links)} links')

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("div", {"class": "art-full"}).find("h1").get_text()
        header = soup.find("div", {"class": "opener"}).get_text()
        article_tag = soup.find("div", {"id": "art-text"})
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
        tag = soup.find("div", {"class": "art-info"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            tag.extract()

    def check_soup(self, soup):
        return True


