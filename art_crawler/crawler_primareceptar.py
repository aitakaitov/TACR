class CrawlerPrimareceptarArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "prima-receptar"
        self.log_path = "log_prima-receptar_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://prima-receptar.cz/"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = False

        self.page = 1
        self.page_max = 653

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article")
        for tag in div_tags:
            author_tag = tag.find('p', {'class': 'author'})
            if author_tag is not None:
                if author_tag.get_text() == 'Komerční sdělení':
                    continue

            a_tag = tag.find("a", {"class": "img"})

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(tag_url)

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}/page/{self.page}'

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        main_div = soup.find("div", {"class": "content"})
        if main_div.find("div", {"class": "preview c"}) is not None:
            main_div.find("div", {"class": "preview c"}).extract()

        if main_div.find('ul', {'class': 'buttons'}) is not None:
            main_div.find('ul', {'class': 'buttons'}).extract()

        tags = main_div.find_all()
        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = main_div.contents
        content_string = ""
        for i in range(len(content)):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return content_string

    def remove_article_heading(self, soup):
        pass
        tag = soup.find("div", {"class": "preview c"})
        if tag is not None:
            tag.extract()
