import urllib.parse


class CrawlerVlastaAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "vlasta"
        self.log_path = "log_vlasta_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://vlasta.kafe.cz/autori/pr-clanek/clanky/"
        self.max_scrolls = 42
        self.max_links = 1000
        self.is_ad = True

        self.page = 1
        self.page_max = 13

    def get_article_urls(self, soup, url):
        links = []

        article_tags = soup.find_all("article")
        for tag in article_tags:
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
            return f'{self.starting_page}?p={self.page}'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("aside", {"class": "article-info-box"})
        if tag is not None:
            tag.extract()

        tags = soup.find("span", {"class": "title"})
        for tag in tags:
            if 'Komerční sdělení' in tag.get_text():
                tag.extract()

        tag = soup.find("div", {"class": "social group"})
        if tag is not None:
            tag.extract()
