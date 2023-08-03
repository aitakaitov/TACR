from library_methods import LibraryMethods
import urllib.parse
import random


class CrawlerAktualneArt:

    def __init__(self):
        self.root_folder = "art_crawler/art_pages"
        self.site_folder = "aktualne"
        self.log_path = "log_aktualne_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.aktualne.cz/prave-se-stalo/"
        self.max_scrolls = 42
        self.max_links = 2000
        self.is_ad = False

    def get_article_urls(self, soup, url):
        links = []
        tags = soup.find("div", {"class": "timeline"}).find_all("div")
        article_tags = []
        valid_classes = ["small-box", "swift-box__wrapper"]
        for div in tags:
            if div.has_attr('class') and any(clss in div['class'] for clss in valid_classes):
                article_tags.append(div)

        for tag in article_tags:
            a_tag = tag.find("a")

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                if "aktualne.cz" in LibraryMethods.strip_url(tag_url):
                    if "sport.aktualne.cz" in LibraryMethods.strip_url(tag_url):
                        rand = random.randint(0, 5)
                        if rand == 0:
                            links.append(urllib.parse.urljoin(url, tag_url))
                    else:
                        links.append(urllib.parse.urljoin(url, tag_url))
                    links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_relevant_text(self, soup):
        try:
            title = soup.find("h1", {"class": "article-title"}).get_text()
        except AttributeError:
            title = ""

        try:
            header = soup.find("div", {"class": "article__perex"}).get_text()
        except AttributeError:
            header = ""

        try:
            article_tag = soup.find("div", {"class": "article__content"})
            tags = article_tag.find_all()
        except AttributeError:
            return title + "\n" + header

        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p":
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = article_tag.contents
        content_string = ""
        for i in range(len(content)):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return title + "\n" + header + "\n" + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "article-subtitle--commercial"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "taglist"})
        if tag is not None:
            tag.extract()

        tag = soup.find('a', {'class': 'author__name'})
        if tag is not None:
            tag.extract()

        tags = soup.find_all('div', {'class': 'article__photo'})
        for tag in tags:
            tag.extract()

        tags = soup.find_all('div', {'class': 'adsense'})
        for tag in tags:
            tag.extract()

    def get_next_page(self, soup, url):
        tag = soup.find("a", {"class": "more-btn"})
        if tag is not None:
            return urllib.parse.urljoin(url, tag.get("href"))
        else:
            tag = soup.find("a", {"class": "listing-nav__btn listing-nav__btn--right"})
            if tag is not None:
                return urllib.parse.urljoin(url, tag.get("href"))
            else:
                return None

    def check_soup(self, soup):
        taglist = soup.find("div", {"class": "taglist"})
        if taglist is not None:
            if "online" in taglist.get_text():
                return False
        return True
