import urllib.parse


class CrawlerLifeeAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "lifee"
        self.log_path = "log_lifee_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.lifee.cz/author/komercni-clanek/"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = True

        self.page = 1
        self.page_max = 56

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article")
        for tag in div_tags:
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
            return f'{self.starting_page}page/{self.page}/'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "article-details"})
        if tag is not None:
            tag.extract()
