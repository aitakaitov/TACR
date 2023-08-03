import json
from hashlib import sha256

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException

from art_crawler.crawler_aktualne import CrawlerAktualneArt
from ad_crawler.crawler_aktualne import CrawlerAktualneAd
from ad_crawler.library_methods import LibraryMethods
from ad_crawler.log import Log
from ad_crawler.persistent_list import PersistentList

from bs4 import BeautifulSoup, Comment

import os
import traceback
import argparse

filename_length = 255

class GenericCrawler():
    def __init__(self, crawler):
        self.crawler = crawler

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--incognito")

        self.log = Log(crawler.log_path)

        ''' Selenium driver for chrome'''
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except WebDriverException:
            self.log.log("[CRAWLER] Chromedriver '" + crawler.chromedriver_path + "' not found, trying .exe")
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except WebDriverException:
                self.log.log("[CRAWLER] No chromedriver found, exiting")
                exit(1)

        ''' Page load timeout'''
        self.driver.set_page_load_timeout(20)

        ''' List of links to visit '''
        self.links_to_visit = PersistentList(crawler.to_visit_file)

        try:
            os.mkdir("./" + crawler.root_folder)
        except OSError:
            self.log.log("[CRAWLER] Pages directory already exists.")

        try:
            os.mkdir("./" + crawler.root_folder + "/" + crawler.site_folder)
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
            self.collect_links(self.crawler.starting_page)
            self.download_links()
        except (WebDriverException, JavascriptException):
            self.log.log("Error loading starting page, will exit.")
            traceback.print_exc()
            return

    def collect_links(self, page):
        self.log.log("Collecting links")
        url = page

        while len(self.links_to_visit) < self.crawler.max_links:
            try:
                html = LibraryMethods.download_page_html(self.driver, url, self.crawler.max_scrolls)
            except WebDriverException:
                break
            soup = BeautifulSoup(html)

            article_urls = self.crawler.get_article_urls(soup, page)

            for article_url in article_urls:
                if article_url not in self.links_to_visit and \
                        len(self.links_to_visit) < self.crawler.max_links:
                    self.links_to_visit.append(article_url)

            url = self.crawler.get_next_page(soup, page)
            if url is None:
                break

    def download_links(self):
        self.log.log("Downloading pages")
        html_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/html"
        relevant_plaintext_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/relevant_plaintext"
        relevant_p_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/relevant_with_p"

        try:
            os.mkdir(html_folder)
            os.mkdir(relevant_plaintext_folder)
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

            if not self.crawler.check_soup(soup):
                continue

            LibraryMethods.filter_html(soup)

            comments = soup.find_all(text=lambda text: isinstance(text, Comment))
            for comment in comments:
                comment.extract()

            filename = sha256(url.encode()).hexdigest() + '.json'

            if len(filename) > filename_length:
                filename = filename[0:filename_length]

            d = {
                'url': url,
                'site': self.crawler.site_folder,
                'data': None,
                'ad': self.crawler.is_ad
            }

            if os.path.exists(html_folder + "/" + filename):
                self.log.log("File " + html_folder + "/" + filename + " exists, skipping")
                continue

            with open(html_folder + "/" + filename, "w+", encoding='utf-8') as f:
                d['data'] = soup.prettify()
                f.write(json.dumps(d))

            self.crawler.remove_article_heading(soup)

            with open(relevant_plaintext_folder + "/" + filename, "w+", encoding='utf-8') as f:
                d['data'] = self.crawler.get_relevant_text(soup)
                f.write(json.dumps(d))

            with open(relevant_p_folder + "/" + filename, "w+", encoding='utf-8') as f:
                LibraryMethods.keep_paragraphs(soup)
                d['data'] = soup.prettify()
                f.write(json.dumps(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site')
    args = vars(parser.parse_args())

    if args['site'].lower() == 'aktualne-art':
        crawler = GenericCrawler(CrawlerAktualneArt())
    elif args['site'].lower() == 'aktualne-ad':
        crawler = GenericCrawler(CrawlerAktualneAd())

    elif args['site'].lower() == 'banky-art':
        # unable to crawl
        exit(0)
    elif args['site'].lower() == 'banky-ad':
        # unable to crawl
        exit(0)

    elif args['site'].lower() == None:
        exit(0)

    crawler.start_crawler()