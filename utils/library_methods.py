import re
import time
from bs4 import BeautifulSoup, Comment, NavigableString
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
    def keep_paragraphs_old(soup: BeautifulSoup):
        tags = soup.find_all('p')
        content = ''
        for tag in tags:
            content += tag.get_text(strip=True) + '\n'

        return content

    @staticmethod
    def keep_paragraphs(soup: BeautifulSoup):
        result_list = []

        p_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for p_tag in p_tags:
            LibraryMethods.process_contents(p_tag, result_list)

        text = '\n'.join(result_list)
        return re.sub('\n+', '\n', text)

    @staticmethod
    def unescape_chars(string):
        return html.unescape(string)

    @staticmethod
    def process_contents(tag, string_list):
        for item in tag.contents:
            if isinstance(item, NavigableString):
                string_list.append(str(item))
            else:
                LibraryMethods.process_contents(item, string_list)
