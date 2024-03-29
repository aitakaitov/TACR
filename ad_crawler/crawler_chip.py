import re
import urllib.parse


class CrawlerChipAd:

    def __init__(self):
        self.root_folder = "ad_pages"
        self.site_folder = "chip"
        self.log_path = "log_chip_ad.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-ad-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.chip.cz/komercni-sdeleni/"
        self.max_scrolls = 42
        self.max_links = 1000
        self.is_ad = True

    def get_article_urls(self, soup, base_url):
        links = []
        main_div = soup.find("div", {"class": "post"})
        div_tags = main_div.find_all("div", {"class": "actuality-wrap actuality-wrap--small"})

        for tag in div_tags:
            a_tag = tag.find("a", {"class": "anotace-image"})

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if urllib.parse.urljoin(base_url, tag_url) not in links:
                links.append(urllib.parse.urljoin(base_url, tag_url))

        return links

    def check_soup(self, soup):
        return True

    def get_next_page(self, soup, url):
        tag = soup.find("div", {"class": "PagerClass"})
        if tag is None:
            return None

        tag = tag.find('a', {'class': 'next'})
        if tag is None:
            return None
        else:
            return tag.get('href')

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("div", {"class": "post-header__title"}).get_text()
        article_tag = soup.find("div", {"class": "post article commercial-post"})
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

        return title + "\n" + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "breadcrumbs"})
        if tag is not None:
            tag.extract()

        tag = soup.find("span", {"class": "post-header__info__author"})
        if tag is not None:
            tag.extract()

        tag = soup.find('div', {'class': 'post article commercial-post'})
        if tag is not None:
            tag = tag.find('div', {'style': 'font-weight:bold;margin-bottom:15px;'})
            tag_text = tag.getText()
            tag_text = re.sub('Komerční sdělení -', '', tag_text)
            tag_text = re.sub('KOMERČNÍ SDĚLENÍ -', '', tag_text)
            tag_text = re.sub('komerční sdělení -', '', tag_text)
            tag_text = re.sub('– KOMERČNÍ SDĚLENÍ –', '', tag_text)
            tag.string = tag_text

        tag = soup.find('div', {'class': 'komercko-ad'})
        if tag is not None:
            tag.extract()
