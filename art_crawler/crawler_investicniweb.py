import urllib.parse


class CrawlerInvesticniwebArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "investicniweb"
        self.log_path = "log_investicniweb_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.investicniweb.cz/co-se-deje"
        self.max_scrolls = 0
        self.max_links = 10000
        self.is_ad = False

        self.page = 1
        self.page_max = 1500

    def get_article_urls(self, soup, url):
        tags = soup.find_all("h2", {"class": "node-title"})
        links = [urllib.parse.urljoin(url, tag.find('a').get('href')) for tag in tags]

        return links

    def check_soup(self, soup):
        auth = soup.find("a", {"class": "article-header-author-link"})
        if auth is not None and "Komerční sdělení" in auth.get_text():
            return False

        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        main_div = soup.find("div", {"class": "article"})

        title = main_div.find("h2", {"class": "article-title title-h1"})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        perex = main_div.find("div", {"class": "article-description"})
        if perex is not None:
            perex = perex.get_text()
        else:
            perex = ''

        article_body = main_div.find("div", {"class": "article-body"})

        tags = article_body.find_all()
        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = article_body.contents
        content_string = ""
        for i in range(len(content)):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return title + '\n' + perex + '\n' + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "article-header-author"})
        if tag is not None:
            tag.extract()

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}?page={self.page}'
