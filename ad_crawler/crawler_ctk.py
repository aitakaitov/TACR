import urllib.parse


class CrawlerCtkAd:

    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "ctk"
        self.log_path = "ctk_log_ad.log"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.ceskenoviny.cz/pr/"
        self.max_scrolls = 1000  # should be about 10k links
        self.max_links = 10000
        self.is_ad = True

    def get_article_urls(self, soup, url):
        links = []
        li_tags = soup.find_all("li", {"class": "list-item"}, recursive=True)
        for tag in li_tags:
            a_tag = tag.find("a")

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if urllib.parse.urljoin(url, tag_url) not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        try:
            title = soup.find("h1", {"itemprop": "name"}).get_text()
        except AttributeError:
            title = ""

        try:
            div_tag = soup.find("div", {"itemprop": "articleBody"})
        except AttributeError:
            return title
        tags = div_tag.find_all()

        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = div_tag.contents
        content_string = ""
        for i in range(len(content) - 2):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return title + "\n" + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "box-article-info"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "box-article-footer"})
        if tag is not None:
            tag.extract()

        tag = soup.find('p', {'class': 'tags'})
        if tag is not None:
            tag.extract()

    def get_next_page(self, soup, url):
        return None

    def check_soup(self, soup):
        return True
