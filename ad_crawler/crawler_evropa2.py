import urllib.parse


class CrawlerEvropa2Ad:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "evropa2"
        self.log_path = "log_evropa2_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.evropa2.cz/clanky/komercni-sdeleni"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = True

        self.page = 1
        self.page_max = 47

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article", {'role': 'article'})
        for tag in div_tags:
            category = tag.find('p')
            if category is not None:
                if 'Komerční sdělení' not in category.get_text():
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
            return f'{self.starting_page}?page={self.page}'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "jsx-3474553550 grid grid--sm"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "jsx-3984645992 row-main"})
        if tag is not None:
            tag.extract()
