import urllib.parse


class CrawlerExtraAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "extra"
        self.log_path = "log_extra_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.extra.cz/autor/komercni-clanek/1"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = True

        self.page = 1
        self.page_max = 37

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'article__list__item'})
        for tag in div_tags:
            a_tag = tag.find("a", {"itemprop": "url"})

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page[:-1]}{self.page}'

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find('header', {'class': 'post__header'})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        main_div = soup.find('div', {'class': 'post__body clearfix'})

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

        return title + '\n' + content_string

    def remove_article_heading(self, soup):
        author_tag = soup.find('a', {'href': '/author/komercni-clanek'})
        if author_tag is not None:
            author_tag.extract()

        ad_tags = soup.find_all('div', {'class': 'cnc-ads cnc-ads--rectangle_480_1'})
        for tag in ad_tags:
            tag.extract()

        garbage = soup.find_all('div', {'class': 'post__gallery px-0'})
        for tag in garbage:
            tag.extract()

        garbage = soup.find_all('div', {'class': 'post__pictures post__pictures--sm'})
        for tag in garbage:
            tag.extract()

