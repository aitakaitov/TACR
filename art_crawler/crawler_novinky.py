class CrawlerNovinkyArt:

    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "novinky"
        self.log_path = "log_novinky_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.novinky.cz/stalo-se"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

    def get_article_urls(self, soup, url):
        links = []
        tags = soup.find_all('article', {'class': 'q_gO q_hn n_gO document-item--default'})
        for tag in tags:
            ad_tag = tag.find('span', {'data-dot': 'atm-label'})
            if ad_tag is not None:
                continue

            a_tag = tag.find('a', {'class': 'c_an q_g2'})

            tag_url = a_tag.get("href")
            if tag_url not in links:
                if tag_url not in links:
                    links.append(tag_url)

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("div", {"data-dot": "ogm-article-header"}).get_text()
        header = soup.find("p", {"data-dot": "ogm-article-perex"}).get_text()
        article_tag = soup.find("article", {"aria-labelledby": "accessibility-article"})
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
        # tag = soup.find("div", {"data-qipojriimlnpiljnkqo": "atm-label"})
        # if tag is not None:
        #     tag.extract()

        tag = soup.find('div', {'data-dot': 'ogm-breadcrumb-navigation'})
        if tag is not None:
            tag.extract()

        tag = soup.find('div', {'data-dot': 'ogm-article-author'})
        if tag is not None:
            tag.extract()

    def get_next_page(self, soup, url):
        expand = soup.find('div', {'class': 'atm-expand-button g_g7 g_hd'})
        if expand is None:
            return None

        url = expand.find('a')
        if url is None:
            return None

        return url.get('href')

    def check_soup(self, soup):
        tag = soup.find('div', {
                                'data-dot': 'ogm-breadcrumb-navigation-item',
                                'data-dot-data': '{"position":"1"}'}
                        )
        if tag is not None:
            if tag.get_text() == 'Komerční články':
                return False

        return True
