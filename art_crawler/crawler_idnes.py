from utils import LibraryMethods


class CrawlerIdnesArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "idnes"
        self.log_path = "log_idnes_art.log"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.idnes.cz/zpravy/archiv/"
        self.max_links = 10000
        self.max_scrolls = 10
        self.is_ad = False
        self.page = 1

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find("div", {"class": "list-art list-art-odklad"}).find_all("div", {"class": "art"})
        for tag in div_tags:
            brisk_tag = tag.find("span", {"class": "brisk"})
            if brisk_tag is not None:
                if "online" in brisk_tag.get_text() or "Komerční sdělení" in brisk_tag.get_text():
                    continue

            prem_tag = tag.find("a", {"class": "premlab"})
            if prem_tag is not None:
                continue

            a_tag = tag.find("a")

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                if "idnes.cz" in LibraryMethods.strip_url(tag_url):
                    links.append(tag_url)

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        return self.starting_page + str(self.page)

    def get_relevant_text(self, soup, keep_paragraphs=True):
        try:
            title = soup.find("div", {"class": "art-full"}).find("h1").get_text()
        except AttributeError:
            title = ""

        try:
            header = soup.find("div", {"class": "opener"}).get_text()
        except AttributeError:
            header = ""

        try:
            article_tag = soup.find("div", {"class": "bbtext"})
            tags = article_tag.find_all()
        except AttributeError:
            return title + header

        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
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
        tag = soup.find("div", {"class": "art-info"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            tag.extract()

    def check_soup(self, soup):
        return True


