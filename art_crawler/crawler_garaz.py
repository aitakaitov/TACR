class CrawlerGarazArt:

    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "garaz"
        self.log_path = "log_garaz_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.garaz.cz/"
        self.max_scrolls = 42
        self.max_links = 100
        self.is_ad = False

    def get_article_urls(self, soup, url):
        links = []
        li_tags = soup.find_all("li", {"class": "b_ca e_h5"})

        for li_tag in li_tags:
            ad_tag = li_tag.find('div', {'class': 'c_fl'})
            if ad_tag is not None:
                ad_text = ad_tag.get_text().lower()
                if 'komerční' in ad_text:
                    continue

            a_tag = li_tag.find('a', {'class': 'b_ap c_fb'})
            if a_tag is not None:
                href = a_tag.get('href')
                if href not in links:
                    links.append(href)

        return links

    def get_relevant_text(self, soup, keep_paragraphs=True):
        title = soup.find("div", {"class": "e_iI"}).get_text()
        header = soup.find("p", {"data-dot": "ogm-article-perex"}).get_text()
        article_tag = soup.find("div", {"class": "c_ey mol-rich-content--for-article"})
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
        for i in range(len(content)):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue

            content_string += "\n" + str(part) + "\n"

        return title + "\n" + header + "\n" + content_string.replace("Seznam advertorial", "")

    def remove_article_heading(self, soup):
        tag = soup.find("span", {"class": "c_e8"})
        if tag is not None:
            tag.extract()

        tag = soup.find("a", {"class": "b_ap e_hj"})
        if tag is not None:
            tag.extract()

    def get_next_page(self, soup, url):
        next_tag = soup.find('a', {'class': 'b_ap b_aM'})
        if next_tag is None:
            return None
        else:
            return next_tag.get('href')

    def check_soup(self, soup):
        return True