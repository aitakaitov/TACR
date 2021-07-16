class Config:
    """
    Contains configuration
    """

    ###
    # Things here can be modified if necessary
    # Path to chromedriver executable has to be specified
    ###

    ''' Path to chromedriver executable '''
    chromedriver_path = "./chromedriver"

    ''' Path to log file '''
    log_path = "log.txt"

    '''Path to pages dir'''
    pages_path = "pages"

    '''Starting url'''
    start_urls = [
        "http://seznam.cz",
        "http://odkazy.seznam.cz/Cestovani/",
        "https://odkazy.seznam.cz/Zpravodajstvi/Casopisy-e-ziny/",
        "https://odkazy.seznam.cz/Kultura-a-umeni/Film/",
        "https://odkazy.seznam.cz/Volny-cas-a-zabava/",
        "https://odkazy.seznam.cz/Sport/",
        "https://odkazy.seznam.cz/Veda-a-technika/"
        "https://odkazy.seznam.cz/Lide-a-spolecnost/"
        "https://odkazy.seznam.cz/Pocitace-a-internet/"
        "https://www.firmy.cz/",
        "http://www.plzen-firmy.cz/"]

    ###
    # Things here can be modified, but may result in inconsistent performance or instability
    ###

    ''' Keywords that terms and conditions links would contain '''
    terms_keywords = ["podminky", "podmínky", "podmínkami", "smluvni", "smluvní", "uzivani", "užívání", "terms", "conditions"]

    ''' Keywords that cookies links would contain '''
    cookies_keywords = ["gdpr", "cookie", "privacy", "ochrana", "soukromi", "soukromí", "zpracovani", "zpracování", "zásady", "zasady", "udaju", "udaje", "údajů", "údaje", "protection", "policy"]

    ''' Timeout for page load '''
    webdriver_timeout = 20

    ''' Max link depth for scraping terms and cookies pages'''
    max_depth = 1

    '''Path to persistent files dir'''
    persistent_path = "persistent"

    '''Persistent to-visit queue name'''
    queue_filename = "links_inqueue.txt"

    '''Persistent visited list name'''
    visited_filename = "visited_links.txt"

    '''Persistent scraped domains list name'''
    scraped_domains_filename = "scraped_domains.txt"

    '''Persistent scraped relevant pages list name'''
    scraped_rel_links_filename = "scraped_links.txt"

    '''Path to file, where domains with no links will be stored'''
    no_links_filename = "nolinks.txt"

    '''Path to file, where domains with no terms will be stored'''
    no_terms_filename = "noterms.txt"

    '''Path to file, where domains with no cookies will be stored'''
    no_cookies_filename = "nocookies.txt"

    '''Maximum scrolls'''
    max_scrolls = 20

    '''Pause between scrolls'''
    scroll_pause = 0.2

    '''Maximum filename length'''
    max_filename_length = 250

    '''Extensions we ignore '''
    blacklisted_extensions = [".jpg", ".png", ".jpeg", ".pdf"]



