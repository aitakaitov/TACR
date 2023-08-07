import urllib.parse


class CrawlerSuperArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "super"
        self.log_path = "log_super_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.super.cz/"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'next-article'})
        for tag in div_tags:
            if 'seznam-advertorial' in tag['class']:
                continue

            a_tag = tag.find('a', {'class': 'illustration'})
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        next_tag = soup.find('h2', {'class': 'more-articles negative-font'})
        if next_tag is not None:
            a_tag = next_tag.find('a')
            if a_tag is not None:
                return urllib.parse.urljoin(url, a_tag.get('href'))

        return None

    def check_soup(self, soup):
        ad_tag = soup.find('div', {'class': 'article-datetime article-block'})
        if ad_tag is not None:
            if 'Seznam Advertorial' in ad_tag.get_text():
                return False
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find('h1', {'class': 'negative-font main-title'})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        perex = soup.find('div', {'class': 'perex clearfix'})
        if perex is not None:
            perex = perex.get_text()
        else:
            perex = ''

        article = soup.find('div', {'class': 'article-block'})

        tags = article.find_all()
        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = article.contents
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
        tag = soup.find("div", {"class": "advertorial-warning"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "article-desc"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "article-datetime article-block"})
        if tag is not None:
            tag.extract()
