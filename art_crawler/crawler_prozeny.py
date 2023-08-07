
root_folder = "art_pages"
site_folder = "prozeny"
log_path = "log_prozeny.log"
chromedriver_path = "./chromedriver"
to_visit_file = "TO_VISIT.PERSISTENT"
visited_file = "VISITED.PERSISTENT"
starting_page = "https://prozeny.cz/sekce"
max_scrolls = 2
filename_length = 255

class CrawlerProzenyArt:
    def __init__(self):
        self.root_folder = "art_pages"
        self.site_folder = "prozeny"
        self.log_path = "log_prozeny_art.log"
        self.chromedriver_path = "./chromedriver"
        self.to_visit_file = self.site_folder + "-art-TO_VISIT.PERSISTENT"
        self.starting_page = "https://prozeny.cz/sekce"
        self.max_scrolls = 42
        self.max_links = 10000
        self.is_ad = False

        self.offset = 0
        self.offset_max = 10000

    def start_crawler(self):
        """
        Starts the crawler from a starting url. The crawler will collect all usable links and then place then in a queue,
        collecting more links as it goes.
        :return:
        """

        # Test if we have no links from previous run
        try:
            self.collect_links(starting_page)
            print(len(self.links_to_visit))
            self.download_links()
        except (WebDriverException, JavascriptException):
            self.log.log("Error loading starting page, will exit.")
            traceback.print_exc()
            return

    def collect_links(self, page):
        self.log.log("Collecting links")
        url = page
        self.driver.get(page)
        for i in range(max_scrolls):
            element = self.driver.find_element_by_class_name("c_A")
            ActionChains(self.driver).click(element).perform()
            time.sleep(4)

        html = LibraryMethods.download_page_html(self.driver, url, 0)
        soup = BeautifulSoup(html)

        li_tags = soup.find_all("li", {"class": "c_eB d_hh"})
        for tag in li_tags:
            ad_tag = tag.find("div", {"class": "e_ib mol-feed-item__top-section"}).find("div", {"class": "e_b7"})
            if ad_tag is not None and "Komerční sdělení" in ad_tag.get_text():
                continue

            a_tag = tag.find("a", recursive=True)

            if a_tag is None:
                continue

            tag_url = a_tag.get("href")
            if urllib.parse.urljoin(page, tag_url) not in self.links_to_visit:
                self.links_to_visit.append(urllib.parse.urljoin(page, tag_url))


    def download_links(self):
        self.log.log("Downloading pages")
        html_folder = root_folder + "/" + site_folder + "/html"
        plaintext_folder = root_folder + "/" + site_folder + "/plaintext"
        p_folder = root_folder + "/" + site_folder + "/plaintext_with_p"
        relevant_p_folder = root_folder + "/" + site_folder + "/relevant_with_p"

        try:
            os.mkdir(html_folder)
            os.mkdir(plaintext_folder)
            os.mkdir(p_folder)
            os.mkdir(relevant_p_folder)
        except FileExistsError:
            pass

        for url in self.links_to_visit:
            self.log.log("Processing " + url)
            try:
                html = LibraryMethods.download_page_html(self.driver, url, 20)
            except WebDriverException:
                continue

            soup = BeautifulSoup(html)
            LibraryMethods.filter_html(soup)
            self.remove_article_heading(soup)

            comments = soup.find_all(text=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            filename = url.replace("/", "_")
            if len(filename) > filename_length:
                filename = filename[0:filename_length]

            with open(html_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(soup.prettify())

            with open(relevant_p_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(self.get_relevant_text(soup))

            with open(plaintext_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(BeautifulSoup(soup.prettify()).getText())

            with open(p_folder + "/" + filename, "w+", encoding='utf-8') as f:
                LibraryMethods.keep_paragraphs(soup)
                f.write(soup.prettify())

    def get_relevant_text(self, soup):
        try:
            title = soup.find("h1", {"class": "c_J c_D"}).get_text()
        except AttributeError:
            title = ""

        try:
            header = soup.find("div", {"class": "d_hc"}).find("p").get_text()
        except AttributeError:
            header = ""

        article_tag = soup.find("article", {"class": "d_g1"})

        temp = article_tag.find("div", {"class": "c_aG"})
        if temp is not None:
            temp.extract()

        tags = article_tag.find_all()

        valid_tags = ["div", "a", "p", "h1", "h2", "h3", "h4", "h5", "strong", "b", "i", "em", "span", "ul", "li"]
        for tag in tags:
            if tag.name == "p":
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

        return title + "\n" + header + "\n" + content_string

    def remove_article_heading(self, soup):
        tag = soup.find("a", {"class": "c_C d_hJ"})
        if tag is not None:
            tag.extract()

        tag = soup.find("a", {"class": "c_C"})
        if tag is not None:
            tag.extract()

        tag = soup.find("span", {"data-dot-data": '{"click":"footer-Reklama"}'})
        if tag is not None:
            tag.extract()


Crawler().start_crawler()