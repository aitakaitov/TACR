import urllib.parse


class CrawlerAktualneAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "aktualne"
        self.log_path = "log_aktualne_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.aktualne.cz/l~b:vs:6f717d9ae0a3f2869c1ab03b4f7/?offset=0"
        self.max_scrolls = 42
        self.max_links = 830
        self.is_ad = True

        self.offset = 0
        self.offset_max = 820

    def get_article_urls(self, soup, url):
        links = []
        article_tags = soup.find_all("div", {"data-ga4-type": "article"})
        for tag in article_tags:
            a_tag = tag.find("a")

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if urllib.parse.urljoin(url, tag_url) not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("h1", {"class": "article-title"}).get_text()
        header = soup.find("div", {"class": "article__perex"}).get_text()
        article_tag = soup.find("div", {"class": "article__content"})
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

        # tags = soup.find_all('div', {'class': 'article__photo'})
        # for tag in tags:
        #     tag.extract()

    def get_next_page(self, soup, url):
        address, num = url.split('=')
        self.offset = self.offset + 20
        if self.offset > self.offset_max:
            return None
        else:
            return f'{address}={self.offset}'
