import urllib.parse


class CrawlerLupaAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "lupa"
        self.log_path = "log_lupa_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.lupa.cz/n/pr-clanek/"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = True

        self.page = 1
        self.page_max = 9

    def get_article_urls(self, soup, url):
        links = []

        a_tags = soup.find_all("a", {'class': 'design-article__heading design-article__link--major design-article__link--default design-article__link'})
        for a_tag in a_tags:
            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}?pi={self.page}'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "design-impressum__cell--default design-impressum__cell--with-separator design-impressum__cell design-impressum__cell--start"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "element-margin-top section"})
        if tag is not None:
            tag.extract()
