import time
from bs4 import BeautifulSoup
from urllib import parse
from crawler.config import Config


class LibraryMethods:
    """
    Static methods
    """

    @staticmethod
    def filter_html(soup: BeautifulSoup):
        """
        Filters tags and their contents from html
        :param soup: Parsed html
        :return: Filtered html
        """
        scripts = soup.find_all("script")
        for tag in scripts:
            tag.decompose()

        iframes = soup.find_all("iframe")
        for tag in iframes:
            tag.decompose()

        link_tags = soup.find_all("link")
        for tag in link_tags:
            tag.decompose()

        metas = soup.find_all("meta")
        for tag in metas:
            tag.decompose()

        styles = soup.find_all("style")
        for tag in styles:
            tag.decompose()

        return soup

    @staticmethod
    def strip_url(url: str):
        """
        Converts URL into its domains (strips protocol, GET and any /...)
        :param url: URL
        :return: Only domains
        """

        stripped_url = parse.urlparse(url).netloc
        if stripped_url[:4] == "www.":
            stripped_url = stripped_url[4:]
        return stripped_url

    @staticmethod
    def download_page_html(driver, url: str):
        """
        Given a driver and URL, downloads the page html code.
        If driver.get(url) times out, it throws WebDriverException.
        :param driver: webdriver
        :param url: url
        :return: page html code
        """
        driver.get(url)
        # takes care of scrolling
        last_height = driver.execute_script("return document.body.scrollHeight")
        scrolls = 1
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(Config.scroll_pause)
            new_height = driver.execute_script("return document.body.scrollHeight")
            scrolls += 1
            if new_height == last_height or scrolls == Config.max_scrolls:
                if driver.execute_script("return document.readyState") == 'complete':
                    break
                else:
                    time.sleep(Config.scroll_pause)
            last_height = new_height

        return driver.page_source
