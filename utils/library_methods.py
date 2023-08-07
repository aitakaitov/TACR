import time
from bs4 import BeautifulSoup, Comment
from urllib import parse
import html


class LibraryMethods:
    """
    Static methods
    """

    char_dict = {}

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
    def download_page_html(driver, url: str, max_scrolls: int):
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
        scrolls = 0
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            scrolls += 1
            if new_height == last_height or scrolls == max_scrolls:
                if driver.execute_script("return document.readyState") == 'complete':
                    break
                else:
                    time.sleep(1)
            last_height = new_height

        return driver.page_source

    @staticmethod
    def download_page_html_timeout(driver, url: str, max_scrolls: int, timeout: int):
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
        scrolls = 0
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            scrolls += 1
            if new_height == last_height or scrolls == max_scrolls:
                if driver.execute_script("return document.readyState") == 'complete':
                    break
                else:
                    time.sleep(1)
            last_height = new_height

        time.sleep(timeout)
        return driver.page_source

    @staticmethod
    def keep_paragraphs(soup: BeautifulSoup):
        tags = soup.find_all()

        for tag in tags:
            if tag.name != "p":
                tag.extract()
            else:
                tag.attrs = {}
                tag.unwrap()

        comments = soup.find_all(text=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()

    @staticmethod
    def unescape_chars(string):
        return html.unescape(string)

