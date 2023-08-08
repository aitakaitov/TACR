import copy
import json
from hashlib import sha256

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException

from ad_crawler.crawler_chip import CrawlerChipAd
from ad_crawler.crawler_cnews import CrawlerCnewsAd
from ad_crawler.crawler_ctk import CrawlerCtkAd
from ad_crawler.crawler_extra import CrawlerExtraAd
from ad_crawler.crawler_idnes import CrawlerIdnesAd
from ad_crawler.crawler_investicniweb import CrawlerInvesticniwebAd
from ad_crawler.crawler_lidovky import CrawlerLidovkyAd
from ad_crawler.crawler_novinky import CrawlerNovinkyAd
from ad_crawler.crawler_primareceptar import CrawlerPrimareceptarAd
from ad_crawler.crawler_super import CrawlerSuperAd
from art_crawler.crawler_aktualne import CrawlerAktualneArt
from ad_crawler.crawler_aktualne import CrawlerAktualneAd
from art_crawler.crawler_chip import CrawlerChipArt
from art_crawler.crawler_cnews import CrawlerCnewsArt
from art_crawler.crawler_ctk import CrawlerCtkArt
from art_crawler.crawler_extra import CrawlerExtraArt
from art_crawler.crawler_garaz import CrawlerGarazArt
from art_crawler.crawler_idnes import CrawlerIdnesArt
from art_crawler.crawler_investicniweb import CrawlerInvesticniwebArt
from art_crawler.crawler_lidovky import CrawlerLidovkyArt
from art_crawler.crawler_novinky import CrawlerNovinkyArt
from art_crawler.crawler_primareceptar import CrawlerPrimareceptarArt
from art_crawler.crawler_super import CrawlerSuperArt

from utils.library_methods import LibraryMethods
from utils.log import Log
from utils.persistent_list import PersistentList

from bs4 import BeautifulSoup, Comment

import os
import traceback
import argparse

os.chdir(os.path.dirname(__file__))

filename_length = 255


class GenericCrawler:
    def __init__(self, crawler):
        self.crawler = crawler

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument('--disable-gpu')

        self.log = Log(crawler.log_path)

        ''' Selenium driver for chrome'''
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
        except WebDriverException as e:
            print(e)

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
            try:
                links = self.crawler.collect_links(self.driver)
                for link in links:
                    self.links_to_visit.append(link)
            except Exception as a:
                print('custom link collection not implemented, using the default one')
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

            try:
                article_urls = self.crawler.get_article_urls(soup, page)
            except Exception as e:
                print('Error collecting links from the page. Trace:')
                print(e)
                continue

            for article_url in article_urls:
                if article_url not in self.links_to_visit and \
                        len(self.links_to_visit) < self.crawler.max_links:
                    self.links_to_visit.append(article_url)

            print(f'Collected {len(self.links_to_visit)} links in total')

            try:
                url = self.crawler.get_next_page(soup, page)
                if url is None:
                    break
            except Exception as e:
                print('Could not get the next page, ending collection')
                print(e)

    def download_links(self):
        self.log.log("Downloading pages")
        html_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/html"
        relevant_plaintext_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/relevant_plaintext"
        fullpage_p_only = self.crawler.root_folder + "/" + self.crawler.site_folder + "/full_only_p"

        try:
            os.makedirs(html_folder)
            os.makedirs(relevant_plaintext_folder)
            os.makedirs(fullpage_p_only)
        except FileExistsError:
            pass

        fails = 0
        for url in self.links_to_visit:
            self.log.log("Processing " + url)
            try:
                html = LibraryMethods.download_page_html(self.driver, url, 20)
            except Exception as e:
                print(f'unable to download {url}')
                fails = fails + 1
                continue

            soup = BeautifulSoup(html)

            if not self.crawler.check_soup(soup):
                print()
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

            soup_backup = copy.copy(soup)

            try:
                self.crawler.remove_article_heading(soup)
            except Exception as e:
                print('Could not remove ad-related stuff from article, skipping')
                print(e)
                continue

            with open(relevant_plaintext_folder + "/" + filename, "w+", encoding='utf-8') as f:
                try:
                    d['data'] = self.crawler.get_relevant_text(soup, keep_paragraphs=False)
                except Exception as e:
                    print('Could not extract relevant text, skipping')
                    print(e)
                    continue

                f.write(json.dumps(d))

            with open(fullpage_p_only + "/" + filename, "w+", encoding='utf-8') as f:
                content = LibraryMethods.keep_paragraphs(soup_backup)
                d['data'] = content
                f.write(json.dumps(d))

        print(f'Failed to download {fails}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site')
    args = vars(parser.parse_args())

    if args['site'].lower() == 'aktualne-art':
        crawler = GenericCrawler(CrawlerAktualneArt())
    elif args['site'].lower() == 'aktualne-ad':
        crawler = GenericCrawler(CrawlerAktualneAd())

    elif args['site'].lower() == 'chip-art':
        crawler = GenericCrawler(CrawlerChipArt())
    elif args['site'].lower() == 'chip-ad':
        crawler = GenericCrawler(CrawlerChipAd())

    elif args['site'].lower() == 'cnews-art':
        crawler = GenericCrawler(CrawlerCnewsArt())
    elif args['site'].lower() == 'cnews-ad':
        crawler = GenericCrawler(CrawlerCnewsAd())

    elif args['site'].lower() == 'ctk-art':
        crawler = GenericCrawler(CrawlerCtkArt())
    elif args['site'].lower() == 'ctk-ad':
        crawler = GenericCrawler(CrawlerCtkAd())

    elif args['site'].lower() == 'garaz-art':
        crawler = GenericCrawler(CrawlerGarazArt())
    elif args['site'].lower() == 'garaz-ad':
        print('Crawling ads in Garaz.cz is not supported')
        exit(1)

    elif args['site'].lower() == 'idnes-art':
        crawler = GenericCrawler(CrawlerIdnesArt())
    elif args['site'].lower() == 'idnes-ad':
        crawler = GenericCrawler(CrawlerIdnesAd())

    elif args['site'].lower() == 'investicniweb-art':
        crawler = GenericCrawler(CrawlerInvesticniwebArt())
    elif args['site'].lower() == 'investicniweb-ad':
        crawler = GenericCrawler(CrawlerInvesticniwebAd())

    elif args['site'].lower() == 'lidovky-art':
        crawler = GenericCrawler(CrawlerLidovkyArt())
    elif args['site'].lower() == 'lidovky-ad':
        crawler = GenericCrawler(CrawlerLidovkyAd())

    elif args['site'].lower() == 'novinky-art':
        crawler = GenericCrawler(CrawlerNovinkyArt())
    elif args['site'].lower() == 'novinky-ad':
        crawler = GenericCrawler(CrawlerNovinkyAd())

    elif args['site'].lower() == 'primareceptar-art':
        crawler = GenericCrawler(CrawlerPrimareceptarArt())
    elif args['site'].lower() == 'primareceptar-ad':
        crawler = GenericCrawler(CrawlerPrimareceptarAd())

    elif args['site'].lower() == 'extra-art':
        crawler = GenericCrawler(CrawlerExtraArt())
    elif args['site'].lower() == 'extra-ad':
        crawler = GenericCrawler(CrawlerExtraAd())

    elif args['site'].lower() == 'super-art':
        crawler = GenericCrawler(CrawlerSuperArt())
    elif args['site'].lower() == 'super-ad':
        crawler = GenericCrawler(CrawlerSuperAd())

    crawler.start_crawler()
