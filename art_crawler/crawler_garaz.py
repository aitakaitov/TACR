import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException
from selenium.webdriver import ActionChains

from library_methods import LibraryMethods
from log import Log
from persistent_list import PersistentList

from bs4 import BeautifulSoup, Comment
import urllib.parse

import os
import traceback
import re

root_folder = "art_pages"
site_folder = "garaz"
log_path = "log_garaz.log"
chromedriver_path = "./chromedriver"
to_visit_file = "TO_VISIT.PERSISTENT"
visited_file = "VISITED.PERSISTENT"
starting_page = "https://www.garaz.cz/"
max_scrolls = 250
filename_length = 255


class Crawler:

    def __init__(self):
        chrome_options = Options()
        #chrome_options.add_argument("--headless")
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
        self.driver.get(page)

        for i in range(max_scrolls):
            time.sleep(5)
            element = self.driver.find_element_by_class_name("c_K c_J")
            ActionChains(self.driver).click(element).perform()

        html = LibraryMethods.download_page_html(self.driver, page, max_scrolls)
        soup = BeautifulSoup(html)

        li_tags = soup.find_all("li", {"class": "c_bV e_f4"})
        for tag in li_tags:

            ad_tag = tag.find("div", {"class": "d_cI"})
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

            filename = re.sub('[^a-zA-Z0-9]', '_', url)

            if len(filename) > filename_length:
                filename = filename[0:filename_length]

            if os.path.exists(html_folder + "/" + filename):
                self.log.log("File " + html_folder + "/" + filename + " exists, skipping")
                continue

            with open(html_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(LibraryMethods.unescape_chars(soup.prettify()))

            with open(relevant_p_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(LibraryMethods.unescape_chars(self.get_relevant_text(soup)))

            with open(plaintext_folder + "/" + filename, "w+", encoding='utf-8') as f:
                f.write(LibraryMethods.unescape_chars(BeautifulSoup(soup.prettify()).getText()))

            with open(p_folder + "/" + filename, "w+", encoding='utf-8') as f:
                LibraryMethods.keep_paragraphs(soup)
                f.write(LibraryMethods.unescape_chars(soup.prettify()))

    def get_relevant_text(self, soup):
        title = soup.find("h1", {"class": "c_c3 c_N"}).get_text()
        header = soup.find("p", {"class": "c_C e_fj"}).get_text()
        article_tag = soup.find("div", {"class": "d_bu mol-rich-content--for-article"})
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

        return title + "\n" + header + "\n" + content_string.replace("Seznam advertorial", "")

    def remove_article_heading(self, soup):
        tag = soup.find("div", {"class": "d_cw"})
        if tag is not None:
            tag.extract()

        tag = soup.find("a", {"class": "e_gA"})
        if tag is not None:
            tag.extract()


Crawler().start_crawler()