class CrawlerLidovkyAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "lidovky"
        self.log_path = "log_lidovky_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.lidovky.cz/pr/sdeleni-komercni/"
        self.max_scrolls = 10
        self.max_links = 10000
        self.is_ad = True

        self.page = 1
        self.page_max = 38

    def get_article_urls(self, soup, url):
        links = []
        tags = soup.find_all("a", {"class": "art-link"})

        for tag in tags:
            href = tag.get('href')
            if href not in links:
                links.append(href)

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        try:
            title = soup.find("h1", {"itemprop": "name headline"}).get_text()
        except AttributeError:
            title = ""
        header = soup.find("div", {"class": "opener"}).get_text()
        article_tag = soup.find("div", {"class": "bbtext"})
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
        return

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}{self.page}'

    def check_soup(self, soup):
        return True
