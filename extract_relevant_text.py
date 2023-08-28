import json
import os
import traceback

from bs4 import BeautifulSoup

from ad_crawler.crawler_forbes import CrawlerForbesAd
from ad_crawler.crawler_investicniweb import CrawlerInvesticniwebAd
from ad_crawler.crawler_lidovky import CrawlerLidovkyAd
from ad_crawler.crawler_primareceptar import CrawlerPrimareceptarAd
from ad_crawler.crawler_super import CrawlerSuperAd
from ad_crawler.crawler_expres import CrawlerExpresAd
from ad_crawler.crawler_idnes import CrawlerIdnesAd
from ad_crawler.crawler_aktualne import CrawlerAktualneAd
from ad_crawler.crawler_ctk import CrawlerCtkAd

from art_crawler.crawler_forbes import CrawlerForbesArt
from art_crawler.crawler_investicniweb import CrawlerInvesticniwebArt
from art_crawler.crawler_lidovky import CrawlerLidovkyArt
from art_crawler.crawler_primareceptar import CrawlerPrimareceptarArt
from art_crawler.crawler_super import CrawlerSuperArt
from art_crawler.crawler_expres import CrawlerExpresArt
from art_crawler.crawler_idnes import CrawlerIdnesArt
from art_crawler.crawler_aktualne import CrawlerAktualneArt
from art_crawler.crawler_ctk import CrawlerCtkArt


def process(_dir, crawlers):
    for crawler, pages_dir in crawlers:
        print(f'-- {pages_dir}')
        pages = os.listdir(os.path.join(_dir, pages_dir, 'html'))
        os.makedirs(os.path.join(_dir, pages_dir, 'relevant_only'), exist_ok=True)
        for page in pages:
            with open(os.path.join(_dir, pages_dir, 'html', page), 'r', encoding='utf-8') as f:
                data = json.loads(f.read())

            html = data['data']
            soup = BeautifulSoup(html)
            crawler.remove_article_heading(soup)
            try:
                data['data'] = crawler.get_relevant_text(soup, keep_paragraphs=False)
            except Exception as e:
                print(f'unable to process {page}')
                traceback.print_exception(e)
                continue

            with open(os.path.join(_dir, pages_dir, 'relevant_only', page), 'w+', encoding='utf-8') as f:
                f.write(json.dumps(data))


def main():
    process('ad_pages', ad_crawlers)
    process('art_pages', art_crawlers)


if __name__ == '__main__':
    art_crawlers = [
        (CrawlerAktualneArt(), 'aktualne'),
        (CrawlerIdnesArt(), 'idnes'),
        #(CrawlerExpresArt(), 'expres'),
        (CrawlerSuperArt(), 'super'),
        (CrawlerForbesArt(), 'forbes'),
        (CrawlerInvesticniwebArt(), 'investicniweb'),
        (CrawlerLidovkyArt(), 'lidovky'),
        (CrawlerPrimareceptarArt(), 'prima-receptar'),
        (CrawlerCtkArt(), 'ctk')
    ]

    ad_crawlers = [
        #(CrawlerAktualneAd(), 'aktualne'),
        #(CrawlerIdnesAd(), 'idnes'),
        #(CrawlerExpresAd(), 'expres'),
        #(CrawlerSuperAd(), 'super'),
        (CrawlerForbesAd(), 'forbes'),
        (CrawlerInvesticniwebAd(), 'investicniweb'),
        (CrawlerLidovkyAd(), 'lidovky'),
        (CrawlerPrimareceptarAd(), 'prima-receptar'),
        (CrawlerCtkAd(), 'ctk')
    ]

    main()
