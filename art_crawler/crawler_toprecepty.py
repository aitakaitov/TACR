import urllib.parse


class CrawlerTopreceptyArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "toprecepty"
        self.log_path = "log_toprecepty_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.toprecepty.cz/clanky/"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.page = 1
        self.page_max = 327

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'b-article__inner'})
        for tag in div_tags:
            auth_tag = tag.find('span', {'class': 'author__name'})
            if auth_tag is not None:
                name = auth_tag.find('a')
                if name is not None:
                    if name.get('href') == '/autor-clanku/komercnisdeleni/':
                        continue

            a_tag = tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}?stranka={self.page}'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("span", {"class": "author__name"})
        if tag is not None:
            tag.extract()
