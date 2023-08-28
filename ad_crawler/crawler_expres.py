import urllib.parse


class CrawlerExpresAd:
    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "expres"
        self.log_path = "log_expres_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.expres.cz/komercni-sdeleni"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = True

        self.page = 1
        self.page_max = 4

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'art'})
        for tag in div_tags:
            a_tag = tag.find('a')
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if 'https://www.expres.cz/' not in tag_url:
                continue

            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find('h1', {'itemprop': 'name headline'})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        opener = soup.find('div', {'class': 'bbtext'})
        if opener is not None:
            opener = title.get_text()
        else:
            opener = ''

        main_div = soup.find('div', {'itemprop': 'articleBody'})

        tags = main_div.find_all()
        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
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

        return title + '\n' + opener + '\n' + content_string

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.starting_page}/{self.page}'

    def check_soup(self, soup):
        return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            tag.extract()
