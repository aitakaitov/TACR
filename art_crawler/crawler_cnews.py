import urllib.parse


class CrawlerCnewsArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "cnews"
        self.log_path = "log_cnews_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://www.cnews.cz/clanky/?pi=1"
        self.max_scrolls = 42
        self.max_links = 1400
        self.is_ad = False

    def get_article_urls(self, soup, base_url):
        links = []
        article_tags = soup.find_all("div", {"class": "design-article--with-image design-article design-tile design-article__category-article"})
        for tag in article_tags:
            a_tag = tag.find("a")

            if a_tag is None:
                continue

            ad_tag = tag.find('a', {'class': 'design-impressum__item--author design-impressum__item'})
            if ad_tag.get_text() == 'Komerční článek':
                continue

            tag_url = a_tag.get("href")
            if urllib.parse.urljoin(base_url, tag_url) not in links:
                links.append(urllib.parse.urljoin(base_url, tag_url))

        return links

    def get_next_page(self, soup, url):
        tag = soup.find('div', {'class': 'pagination'})
        if tag is None:
            return None

        tag = tag.find('a', {'class': 'next arrow'})
        if tag is None:
            return None
        else:
            return tag.get('href')

    def check_soup(self, soup):
        return True

    def get_relevant_text(self, soup):
        title = soup.find("h1", {"class": "design-article--with-image design-article design-tile design-article__category-article"})
        if title is None:
            title = soup.find('h1', {'class': 'design-heading--level-1 design-heading'})
        if title is not None:
            title = title.get_text()
        else:
            title = ''

        perex = soup.find('div', {'class': 'o-article__perex-text'}).get_text()
        div_tag = soup.find("div", {"class": "layout-article-content"})

        tags = div_tag.find_all()
        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p":
                tag.attrs = {}
            elif tag.name in valid_tags:
                tag.unwrap()
            else:
                tag.extract()

        content = div_tag.contents
        content_string = ""
        for i in range(len(content) - 2):

            part = content[i]
            if len(part) == 0:
                continue
            if str(part).isspace():
                continue
            if "kk-star-ratings" in str(part):
                continue

            content_string += "\n" + str(part) + "\n"

        return title + "\n" + perex + '\n' + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("li", {"class": "entry-category"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "pr-article-detail-info-bar__category"})
        if tag is not None:
            tag.extract()

        tags = soup.find_all("div", {"class": "breadcrumb clear design-breadrumbs"})
        for tag in tags:
            tag.extract()

        tag = soup.find("div", {"class": "design-impressum__item--author design-impressum__item"})
        if tag is not None:
            tag.extract()

        tags = soup.find_all('a', {'class': 'design-label--default design-label'})
        for tag in tags:
            tag.extract()

        tags = soup.find_all('div', {'class': 'ad-detail'})
        for tag in tags:
            tag.extract()

        tags = soup.find_all('div', {'data-advert-marker': 'reklama'})
        for tag in tags:
            tag.extract()
