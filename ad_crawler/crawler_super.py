import urllib.parse


class CrawlerSuperAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "super"
        self.log_path = "log_super_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.super.cz/seznam-advertorial"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = True

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'next-article seznam-advertorial'})
        for tag in div_tags:
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
                if a_tag.get('href') == '/':
                    return None
                else:
                    return urllib.parse.urljoin(url, a_tag.get('href'))

        return None

    def check_soup(self, soup):
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
