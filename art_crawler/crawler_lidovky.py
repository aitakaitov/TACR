class CrawlerLidovkyArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "lidovky"
        self.log_path = "log_lidovky_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.lidovky.cz/archiv/"
        self.max_scrolls = 10
        self.max_links = 10000
        self.is_ad = False

        self.page = 1
        self.page_max = 1_000_000

    def get_article_urls(self, soup, url):
        links = []
        tags = soup.find_all("div", {"class": "art"})

        for tag in tags:
            brisk = tag.find('span', {'class': 'brisk'})
            if brisk is not None:
                if brisk.get_text() == 'Komerční sdělení' or 'online' in brisk.get_text().lower():
                    continue

            premlab = tag.find('a', {'class': 'premlab'})
            if premlab is not None:
                continue

            a_tag = tag.find('a', {'class': 'art-link'})
            href = a_tag.get('href')
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

    def check_soup(self, soup):
        return True

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}{self.page}'
