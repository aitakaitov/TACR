import json
import os
from bs4 import BeautifulSoup

from ad_crawler import crawler_forbes as ad_crawler_forbes
from ad_crawler import crawler_investicniweb as ad_crawler_investicniweb
from ad_crawler import crawler_lidovky as ad_crawler_lidovky
from ad_crawler import crawler_novinky as ad_crawler_novinky
from ad_crawler import crawler_primareceptar as ad_crawler_primareceptar
from ad_crawler import crawler_super as ad_crawler_super
from ad_crawler import crawler_expres as ad_crawler_expres
from ad_crawler import crawler_idnes as ad_crawler_idnes
from ad_crawler import crawler_aktualne as ad_crawler_aktualne

from art_crawler import crawler_forbes as art_crawler_forbes
from art_crawler import crawler_investicniweb as art_crawler_investicniweb
from art_crawler import crawler_lidovky as art_crawler_lidovky
from art_crawler import crawler_novinky as art_crawler_novinky
from art_crawler import crawler_primareceptar as art_crawler_primareceptar
from art_crawler import crawler_super as art_crawler_super
from art_crawler import crawler_expres as art_crawler_expres
from art_crawler import crawler_idnes as art_crawler_idnes
from art_crawler import crawler_aktualne as art_crawler_aktualne


def process(_dir, crawlers):
    for crawler, pages_dir in crawlers:
        pages = os.listdir(os.path.join(_dir, pages_dir, 'html'))
        os.mkdir(os.path.join(_dir, pages_dir, 'relevant_only'))
        for page in pages:
            with open(os.path.join(_dir, pages_dir, 'html', page), 'r', encoding='utf-8') as f:
                data = json.loads(f.read())

            html = data['data']
            soup = BeautifulSoup(html)
            crawler.remove_article_heading(soup)
            data['data'] = crawler.get_relevant_text(soup, keep_paragraphs=False)

            with open(os.path.join(_dir, pages, 'relevant_only', page), 'w+', encoding='utf-8') as f:
                f.write(json.dumps(data))


def main():
    process('ad_pages', ad_crawlers)
    process('art_pages', art_crawlers)


if __name__ == '__main__':
    art_crawlers = [
        (art_crawler_aktualne, 'aktualne'),
        (art_crawler_idnes, 'idnes'),
        (art_crawler_expres, 'expres'),
        (art_crawler_super, 'super'),
        (art_crawler_forbes, 'forbes'),
        (art_crawler_investicniweb, 'investicniweb'),
        (art_crawler_lidovky, 'lidovky'),
        (art_crawler_novinky, 'novinky'),
        (art_crawler_primareceptar, 'primareceptar')
    ]

    ad_crawlers = [
        (ad_crawler_aktualne, 'aktualne'),
        (ad_crawler_idnes, 'idnes'),
        (ad_crawler_expres, 'expres'),
        (ad_crawler_super, 'super'),
        (ad_crawler_forbes, 'forbes'),
        (ad_crawler_investicniweb, 'investicniweb'),
        (ad_crawler_lidovky, 'lidovky'),
        (ad_crawler_novinky, 'novinky'),
        (ad_crawler_primareceptar, 'primareceptar')
    ]

    main()