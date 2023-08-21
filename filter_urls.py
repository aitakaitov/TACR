import os
import json



def analyze(tld, _dir):
    is_ads = tld == 'ad_pages'
    files = os.listdir(os.path.join(tld, _dir, 'p_only_sanitized'))

    required_url = domain_url_dict[_dir]

    wrong_url_files = []
    wrong_urls = []

    if is_ads:
        keyword_occurrences = {}
    
    for file in files:
        with open(os.path.join(tld, _dir, 'p_only_sanitized', file), 'r') as f:
            art = json.loads(f.read())
        
        if required_url not in art['url']: 
            wrong_url_files.append(file) 
            wrong_urls.append(art['url'])
        
        if not is_ads:
            continue
        
        text = art['data'].lower()
        for kw in ad_keywords:
            if kw in text:
                if kw not in keyword_occurrences.keys():
                    keyword_occurrences[kw] = {
                        'files': [file],
                        'urls': [art['url']]
                    }
                else:
                    keyword_occurrences[kw]['files'].append(file)
                    keyword_occurrences[kw]['urls'].append(art['url'])
    
    if not is_ads:
        return {
            'wrong_urls': wrong_urls,
            'wrong_url_files': wrong_url_files
        }
    else:
        return {
            'wrong_urls': wrong_urls,
            'wrong_url_files': wrong_url_files,
            'keyword_occurrences': keyword_occurrences
        } 


def main():
    tlds = ['ad_pages']#, 'art_pages']
    dirs = [k for k, v in domain_url_dict.items()]

    results = {}
    for tld in tlds:
        print(f'--- processing {tld}')
        results[tld] = {}
        for _dir in dirs:
            print(f'processing domain {_dir}')
            results[tld][_dir] = analyze(tld, _dir)

    with open('url_data_ad.json', 'w+', encoding='utf-8') as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    domain_url_dict = {
        'extra': 'https://www.extra.cz',
        'aktualne': 'aktualne.cz/',
        'ahaonline': 'https://www.ahaonline.cz/',
        'drbna': 'drbna.cz/',
        'expres': 'https://www.expres.cz/', #'https://www.expres.cz/komercni-sdeleni/',
        'ctk': 'https://www.ceskenoviny.cz/', #'https://www.ceskenoviny.cz/pr/',
        'lidovky': 'https://www.lidovky.cz/', #'https://www.lidovky.cz/pr/',
        'prima-receptar': 'https://prima-receptar.cz/',
        'super': 'https://www.super.cz/',
        'idnes': 'idnes.cz/', #'https://sdeleni.idnes.cz/',
        'forbes': 'forbes.cz/',
        'chip': 'https://www.chip.cz/',
        'investicniweb': 'https://www.investicniweb.cz/',
        'tiscali': 'tiscali.cz/',
        'cnews': 'https://www.cnews.cz/', #'https://www.cnews.cz/pr-clanky/',
        'toprecepty': 'https://www.toprecepty.cz/',
        'emimino': 'https://www.emimino.cz/',
        'forum24': 'https://www.forum24.cz/',
        'lifee': 'https://www.lifee.cz/',
        'evropa2': 'https://www.evropa2.cz/' #'https://www.evropa2.cz/clanky/komercni-sdeleni/'
    }

    ad_keywords = [
        'komerční sdělení',
        'reklamní sdělení',
        'komerční prezentace', 
        'komerční článek', 
        'reklamní článek', 
        'pr článek', 
        'pr příspěvek', 
        'pr sdělení',
        'reklamní prezentace',
        'komerční příspěvek',
        'advertorial',
        'prezentace klienta',
        '- komerční sdělení -',
        'advoice',
        'seznam advertorial',
        'inzerce'
    ]

    main()