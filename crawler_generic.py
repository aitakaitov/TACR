import copy
import json
from hashlib import sha256

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import JavascriptException

from ad_crawler.crawler_ahaonline import CrawlerAhaonlineAd
from ad_crawler.crawler_chip import CrawlerChipAd
from ad_crawler.crawler_cnews import CrawlerCnewsAd
from ad_crawler.crawler_ctk import CrawlerCtkAd
from ad_crawler.crawler_emimino import CrawlerEmiminoAd
from ad_crawler.crawler_evropa2 import CrawlerEvropa2Ad
from ad_crawler.crawler_expres import CrawlerExpresAd
from ad_crawler.crawler_extra import CrawlerExtraAd
from ad_crawler.crawler_forbes import CrawlerForbesAd
from ad_crawler.crawler_forum24 import CrawlerForum24Ad
from ad_crawler.crawler_idnes import CrawlerIdnesAd
from ad_crawler.crawler_investicniweb import CrawlerInvesticniwebAd
from ad_crawler.crawler_lidovky import CrawlerLidovkyAd
from ad_crawler.crawler_lifee import CrawlerLifeeAd
from ad_crawler.crawler_primareceptar import CrawlerPrimareceptarAd
from ad_crawler.crawler_super import CrawlerSuperAd
from ad_crawler.crawler_vlasta import CrawlerVlastaAd
from art_crawler.crawler_ahaonline import CrawlerAhaonlineArt
from art_crawler.crawler_aktualne import CrawlerAktualneArt
from ad_crawler.crawler_aktualne import CrawlerAktualneAd
from art_crawler.crawler_chip import CrawlerChipArt
from art_crawler.crawler_cnews import CrawlerCnewsArt
from art_crawler.crawler_ctk import CrawlerCtkArt
from art_crawler.crawler_emimino import CrawlerEmiminoArt
from art_crawler.crawler_evropa2 import CrawlerEvropa2Art
from art_crawler.crawler_expres import CrawlerExpresArt
from art_crawler.crawler_extra import CrawlerExtraArt
from art_crawler.crawler_forbes import CrawlerForbesArt
from art_crawler.crawler_forum24 import CrawlerForum24Art
from art_crawler.crawler_garaz import CrawlerGarazArt
from art_crawler.crawler_idnes import CrawlerIdnesArt
from art_crawler.crawler_investicniweb import CrawlerInvesticniwebArt
from art_crawler.crawler_irozhlas import CrawlerIrozhlasArt
from art_crawler.crawler_lidovky import CrawlerLidovkyArt
from art_crawler.crawler_lifee import CrawlerLifeeArt
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
                self.links_to_visit.extend(links)

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
        sanitized_p_only_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/p_only_sanitized"
        p_only_folder = self.crawler.root_folder + "/" + self.crawler.site_folder + "/p_only"

        try:
            os.makedirs(html_folder)
            os.makedirs(sanitized_p_only_folder)
            os.makedirs(p_only_folder)
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
                print('ad')
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

            with open(sanitized_p_only_folder + "/" + filename, "w+", encoding='utf-8') as f:
                content = LibraryMethods.keep_paragraphs(soup)
                d['data'] = content
                f.write(json.dumps(d))

            # with open(relevant_plaintext_folder + "/" + filename, "w+", encoding='utf-8') as f:
            #     try:
            #         d['data'] = self.crawler.get_relevant_text(soup, keep_paragraphs=False)
            #     except Exception as e:
            #         print('Could not extract relevant text, skipping')
            #         print(e)
            #         continue
            #
            #     f.write(json.dumps(d))

            with open(p_only_folder + "/" + filename, "w+", encoding='utf-8') as f:
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
        exit(-1)

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

    elif args['site'].lower() == 'cnn-iprima-art':
        exit(-1)
    elif args['site'].lower() == 'cnn-iprima-ad':
        exit(-1)

    elif args['site'].lower() == 'irozhlas-art':
        crawler = GenericCrawler(CrawlerIrozhlasArt())
    elif args['site'].lower() == 'irozhlas-ad':
        print('Crawling ads in Garaz.cz is not supported')
        exit(1)

    elif args['site'].lower() == 'forbes-art':
        crawler = GenericCrawler(CrawlerForbesArt())
    elif args['site'].lower() == 'forbes-ad':
        crawler = GenericCrawler(CrawlerForbesAd())

    elif args['site'].lower() == 'ahaonline-art':
        crawler = GenericCrawler(CrawlerAhaonlineArt())
    elif args['site'].lower() == 'ahaonline-ad':
        crawler = GenericCrawler(CrawlerAhaonlineAd())

    elif args['site'].lower() == 'forum24-art':
        crawler = GenericCrawler(CrawlerForum24Art())
    elif args['site'].lower() == 'forum24-ad':
        crawler = GenericCrawler(CrawlerForum24Ad())

    elif args['site'].lower() == 'emimino-art':
        crawler = GenericCrawler(CrawlerEmiminoArt())
    elif args['site'].lower() == 'emimino-ad':
        crawler = GenericCrawler(CrawlerEmiminoAd())

    elif args['site'].lower() == 'vlasta-art':
        print('crawling articles in vlasta.cz is not supported')
    elif args['site'].lower() == 'vlasta-ad':
        crawler = GenericCrawler(CrawlerVlastaAd())

    elif args['site'].lower() == 'evropa2-art':
        crawler = GenericCrawler(CrawlerEvropa2Art())
    elif args['site'].lower() == 'evropa2-ad':
        crawler = GenericCrawler(CrawlerEvropa2Ad())

    elif args['site'].lower() == 'lifee-art':
        crawler = GenericCrawler(CrawlerLifeeArt())
    elif args['site'].lower() == 'lifee-ad':
        crawler = GenericCrawler(CrawlerLifeeAd())

    elif args['site'].lower() == 'expres-art':
        crawler = GenericCrawler(CrawlerExpresArt())
    elif args['site'].lower() == 'expres-ad':
        crawler = GenericCrawler(CrawlerExpresAd())

    crawler.start_crawler()
