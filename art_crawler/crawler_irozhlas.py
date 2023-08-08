import urllib.parse
import datetime


class CrawlerIrozhlasArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "irozhlas"
        self.log_path = "log_irozhlas_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.irozhlas.cz/zpravy-archiv/2023-08-07"
        self.max_scrolls = 42
        self.max_links = 200 #10000
        self.is_ad = False

        self.current_day = 1
        self.max_days = 500
        self.current_date = datetime.datetime(2023, 8, 7)

    def get_date_string(self):
        return self.current_date.strftime('%Y-%m-%d')

    def get_article_urls(self, soup, url):
        links = []

        div_tags = soup.find_all("article", {'role': 'article'})
        for tag in div_tags:
            a_tag = tag.find('a', {'class': 'b-article__link'})
            if a_tag is None:
                continue

            tag_tag = tag.find('span', {'class': 'text-uppercase text-bold text-red'})
            if tag_tag is not None:
                tag_text = tag.get_text().lower()
                if 'reklama' in tag_text or 'komerční' in tag_text or 'pr ' in tag_text or 'reklamní' in tag_text:
                    continue

            tag_tag = tag.find('span', {'class': 'hide--m'})
            if tag_tag is not None:
                tag_text = tag.get_text().lower()
                if 'reklama' in tag_text or 'komerční' in tag_text or 'pr ' in tag_text or 'reklamní' in tag_text:
                    continue

            tag_url = a_tag.get("href")
            if tag_url not in links:
                links.append(urllib.parse.urljoin(url, tag_url))

        return links

    def get_next_page(self, soup, url):
        self.current_day += 1
        self.current_date -= datetime.timedelta(days=1)
        if self.current_day > self.max_days:
            return None
        else:
            return f'{self.starting_page[:-10]}{self.get_date_string()}'

    def check_soup(self, soup):
        # I havent seen any commercial articles there
        return True

    def get_relevant_text(self, soup, keep_paragraphs=True):
        article = soup.find('article', {'role': 'article'})

        tags = article.find_all()
        valid_tags = ["a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p" and keep_paragraphs:
                tag.attrs = {}
            elif tag.name in valid_tags:
                new_tag = soup.new_tag('asdf')
                new_tag.string = '\n'
                tag.replace_with(new_tag)
                new_tag.unwrap()
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

        return content_string

    def remove_article_heading(self, soup):
        tag = soup.find("span", {"class": "meta__text"})
        if tag is not None:
            tag.extract()

        tag = soup.find("p", {"class": "meta meta--right meta--big"})
        if tag is not None:
            tag.extract()
