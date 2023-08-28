import urllib.parse


class CrawlerExpresArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "expres"
        self.log_path = "log_expres_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.expres.cz/zpravy/2"
        self.base_url = 'https://www.expres.cz/zpravy/'
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.page = 2
        self.page_max = 392

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("div", {'class': 'art'})
        for tag in div_tags:
            prem_tag = tag.find('a', {'class': 'premlab'})
            if prem_tag is not None:
                continue

            a_tag = tag.find('a', {'class': 'art-link'})
            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if 'https://www.expres.cz/' not in tag_url:
                continue

            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.page += 1
        if self.page > self.page_max:
            return None
        else:
            return f'{self.base_url}/{self.page}'

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find('h1', {'itemprop': 'name headline'})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        opener = soup.find('div', {'itemprop': 'description'})
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

    def check_soup(self, soup):
        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            return False
        else:
            return True

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"id": "komercni-sdeleni"})
        if tag is not None:
            tag.extract()
