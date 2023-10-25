from importlib import import_module
import os
from bs4 import BeautifulSoup
import json


domain_crawler_dict = {
    'drbna': ('crawler.crawler_drbna', 'CrawlerDrbna'),
    'aktualne': ('crawler.crawler_aktualne', 'CrawlerAktualne'),
    'ahaonline': ('crawler.crawler_ahaonline', 'CrawlerAhaonline'),
    'chip': ('crawler.crawler_chip', 'CrawlerChip'),
    'cnews': ('crawler.crawler_cnews', 'CrawlerCnews'),
    'ctk': ('crawler.crawler_ctk', 'CrawlerCtk'),
    'emimino': ('crawler.crawler_emimino', 'CrawlerEmimino'),
    'evropa2': ('crawler.crawler_evropa2', 'CrawlerEvropa2'),
    'expres': ('crawler.crawler_expres', 'CrawlerExpres'),
    'extra': ('crawler.crawler_extra', 'CrawlerExtra'),
    'forbes': ('crawler.crawler_forbes', 'CrawlerForbes'),
    'forum24': ('crawler.crawler_forum24', 'CrawlerForum24'),
    'idnes': ('crawler.crawler_idnes', 'CrawlerIdnes'),
    'investicniweb': ('crawler.crawler_investicniweb', 'CrawlerInvesticniweb'),
    'lidovky': ('crawler.crawler_lidovky', 'CrawlerLidovky'),
    'lifee': ('crawler.crawler_lifee', 'CrawlerLifee'),
    'lupa': ('crawler.crawler_lupa', 'CrawlerLupa'),
    'novinky': ('crawler.crawler_novinky', 'CrawlerNovinky'),
    'primareceptar': ('crawler.crawler_primareceptar', 'CrawlerPrimareceptar'),
    'super': ('crawler.crawler_super', 'CrawlerSuper'),
    'tiscali': ('crawler.crawler_tiscali', 'CrawlerTiscali'),
    'toprecepty': ('crawler.crawler_toprecepty', 'CrawlerToprecepty'),
    'vlasta': ('crawler.crawler_vlasta', 'CrawlerVlasta'),

}


def instantiate_crawler(domain, prefix):
    clss = getattr(
        import_module(f'{prefix}_{domain_crawler_dict[domain][0]}'),
        f'{domain_crawler_dict[domain][1]}{"Ad" if prefix == "ad" else "Art"}'
    )
    return clss()


def process_file(file, crawler) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    soup = BeautifulSoup(data['data'])
    crawler.remove_article_heading(soup)

    data['data'] = soup.prettify()
    return json.dumps(data)


def write_text(file, text):
    with open(file, 'w+', encoding='utf-8') as f:
        f.write(text)


def main():
    # ad pages
    print('-- ad_pages')
    for domain in os.listdir('ad_pages'):
        print(domain)
        crawler = instantiate_crawler(domain, 'ad')
        os.makedirs(f'ad_pages/{domain}/html_sanitized', exist_ok=True)
        # process each HTML file
        for file in os.listdir(f'ad_pages/{domain}/html'):
            sanitized = process_file(f'ad_pages/{domain}/html/{file}', crawler)
            write_text(f'ad_pages/{domain}/html_sanitized/{file}', sanitized)

    # art pages
    print('-- art_pages')
    for domain in os.listdir('art_pages'):
        print(domain)
        crawler = instantiate_crawler(domain, 'art')
        os.makedirs(f'art_pages/{domain}/html_sanitized', exist_ok=True)
        # process each HTML file
        for file in os.listdir(f'art_pages/{domain}/html'):
            sanitized = process_file(f'art_pages/{domain}/html/{file}', crawler)
            write_text(f'art_pages/{domain}/html_sanitized/{file}', sanitized)


if __name__ == '__main__':
    main()


