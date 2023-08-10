import urllib.parse


class CrawlerExpresArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "expres"
        self.log_path = "log_expres_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.expres.cz/zpravy/2"
        self.base_url = 'https://www.expres.cz/zpravy/'
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.page = 2
        self.page_max = 392

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'art'})
        for tag in div_tags:
            prem_tag = tag.find('a', {'class': 'premlab'})
            if prem_tag is not None:
                continue

            a_tag = tag.find('a', {'class': 'art-link'})
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if 'https://www.expres.cz/' not in tag_url:
                continue

            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.base_url}/{self.page}'

    def check_soup(self, soup):
        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            return False
        else:
            return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            tag.extract()
