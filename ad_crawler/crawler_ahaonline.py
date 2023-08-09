class CrawlerAhaonlineAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "ahaonline"
        self.log_path = "log_ahaonline_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.ahaonline.cz/kategorie/7822/"
        self.max_scrolls = 42
        self.max_links = 1000
        self.is_ad = True

        self.page = 1
        self.page_max = 23

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article", {'class': 'inner_8 behavSource'})
        for tag in div_tags:
            a_tag = tag.find('a', {'class': 'readMore floatRight'})
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")

            if 'https://www.ahaonline.cz/' not in tag_url:
                continue
            else:
                pass

            if tag_url not in links:
                links.append(tag_url)
            else:
                pass

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
        tag = soup.find("div", {"class": "articleMeta"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "article-keywords"})
        if tag is not None:
            tag.extract()
