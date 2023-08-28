class CrawlerForbesAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "forbes"
        self.log_path = "log_forbes_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://forbes.cz/tag/advoice/page/1/"
        self.max_scrolls = 42
        self.max_links = 1000
        self.is_ad = True

        self.page = 1
        self.page_max = 21

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'article-card'})
        for tag in div_tags:
            a_tag = tag.find('a', {'class': 'article-card__title-wrapper h-align-items--start h-text-decoration--none h-color--top'})
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
            return f'{self.starting_page[:-2]}{self.page}/'

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find('h1', {'class': 'header__title header__title--colored'})
        title = '' if title is None else title.get_text()

        article = soup.find('div', {'class': 'gutenberg-content large-first-p postContent'})
        tags = article.find_all()
        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
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

        return title + '\n' + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "article-footer__categories-wrapper"})
        if tag is not None:
            tag.extract()

        tag = soup.find("a", {"class": "label label--line"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "brandvoice-author-card__author-wrapper"})
        if tag is not None:
            tag.extract()

        tag = soup.find('div', {'class': 'brandvoice-modal'})
        if tag is not None:
            tag.extract()
