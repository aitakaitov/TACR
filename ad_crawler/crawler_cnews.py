from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException

from library_methods import LibraryMethods
from log import Log
from persistent_list import PersistentList

from bs4 import BeautifulSoup
import urllib.parse

import os
import re
import traceback

root_folder = "ad_pages"
site_folder = "cnews"
log_path = "log_cnews.log"
chromedriver_path = "./chromedriver"
to_visit_file = "TO_VISIT.PERSISTENT"
visited_file = "VISITED.PERSISTENT"
starting_page = "https://www.cnews.cz/komercni-clanek/"
max_scrolls = 2
filename_length = 255


class Crawler:

    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--incognito")

        self.log = Log(log_path)

        ''' Selenium driver for chrome'''
        try:
            self.driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options)
        except WebDriverException:
            self.log.log("[CRAWLER] Chromedriver '" + chromedriver_path + "' not found, trying .exe")
            try:
                self.driver = webdriver.Chrome(executable_path=chromedriver_path + ".exe", options=chrome_options)
            except WebDriverException:
                self.log.log("[CRAWLER] No chromedriver found, exiting")
                exit(1)

        ''' Page load timeout'''
        self.driver.set_page_load_timeout(20)

        ''' List of links to visit '''
        self.links_to_visit = PersistentList(to_visit_file)

        ''' List of visited links '''
        #self.visited_links = PersistentList(visited_file)

        try:
            os.mkdir("./" + root_folder)
        except OSError:
            self.log.log("[CRAWLER] Pages directory already exists.")

        try:
            os.mkdir("./" + root_folder + "/" + site_folder)
        except OSError:
            pass

    def start_crawler(self):
        """
        Starts the crawler from a starting url. The crawler will collect all usable links and then place then in a queue,
        collecting more links as it goes.
        :return:
        """

        # Test if we have no links from previous run
        try:
            self.collect_links(starting_page)
            self.download_links()
        except (WebDriverException, JavascriptException):
            self.log.log("Error loading starting page, will exit.")
            traceback.print_exc()
            return

    def collect_links(self, page):
        self.log.log("Collecting links")
        url = page

        for i in range(max_scrolls):
            try:
                html = LibraryMethods.download_page_html(self.driver, url, max_scrolls)
            except WebDriverException:
                break
            soup = BeautifulSoup(html)

            article_tags = soup.find_all("div", {"class": "td_module_10 td_module_wrap td-animation-stack"})
            for tag in article_tags:
                a_tag = tag.find("h3", {"class": "entry-title td-module-title"}).find("a")

                if a_tag is None:
                    continue

                tag_url = a_tag.get("href")
                if urllib.parse.urljoin(page, tag_url) not in self.links_to_visit:
                    self.links_to_visit.append(urllib.parse.urljoin(page, tag_url))

            url = page + "/page/" + str(i + 2)

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

            filename = url.replace("/", "_")
            parts = filename.split("-")
            filename = ""
            for part in parts[0:len(parts) - 1]:
                filename += part + "-"

            if len(filename) > filename_length:
                filename = filename[0:filename_length]

            if os.path.exists(html_folder + "/" + filename):
                self.log.log("File " + html_folder + "/" + filename + " exists, skipping")
                continue

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
        title = soup.find("h1", {"class": "entry-title"}).get_text()
        div_tag = soup.find("div", {"class": "td-post-content"})

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

        return title + "\n" + content_string


    def remove_article_heading(self, soup):
        tag = soup.find("li", {"class": "entry-category"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "td-post-author-name"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "td-post-source-tags"})
        if tag is not None:
            tag.extract()

        tag = soup.find("div", {"class": "td-author-name vcard author"})
        if tag is not None:
            tag.extract()


Crawler().start_crawler()