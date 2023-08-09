class CrawlerEmiminoAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "emimino"
        self.log_path = "log_emimino_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.emimino.cz/clanky/pr/"
        self.max_scrolls = 42
        self.max_links = 1000
        self.is_ad = True

        self.page = 1
        self.page_max = 18

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article", {'role': 'article'})
        for tag in div_tags:
            a_tag = tag.find('a')
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
            return f'{self.starting_page}strankovani/{self.page}/'

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        # getting only the relevant text is probably useless, so we wont bother with it for now
        return

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
        tag = soup.find("ul", {"class": "about-info"})
        if tag is not None:
            tag.extract()
