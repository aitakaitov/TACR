from bs4 import BeautifulSoup, NavigableString
import re


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


def process_contents(tag, string_list):
    for item in tag.contents:
        if isinstance(item, NavigableString):
            string_list.append(str(item))
        else:
            process_contents(item, string_list)


def keep_paragraphs(soup: BeautifulSoup):
    result_list = []

    p_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for p_tag in p_tags:
        process_contents(p_tag, result_list)

    text = '\n'.join(result_list)
    return re.sub('\n+', '\n', text)


def trim_text_start_length(text, trim_length):
    tag_texts = text.split('\n')
    start = -1
    for i, tag in enumerate(tag_texts):
        if len(tag.split()) > trim_length:
            start = i
            break

    new_text = ''
    for i in range(start, len(tag_texts)):
        new_text += tag_texts[i]

    return new_text


def trim_text(text, trim_type='start_length', trim_length=15):
    if trim_type is None:
        return text
    elif trim_type == 'start_length':
        return trim_text_start_length(text, trim_length)
    else:
        print(f'{trim_type} not valid')
        exit()


def process_html_full(html, trim_type='start_length', trim_length=15):
    soup = BeautifulSoup(html)
    soup = filter_html(soup)
    text = keep_paragraphs(soup)
    trimmed = trim_text(text, trim_type, trim_length)
    return trimmed


def parse_xpath(xp):
    path_nodes = xp.split('<')
    path_nodes.reverse()
    path_nodes = [n.strip() for n in path_nodes]

    path = []
    for node in path_nodes:
        split = node.split('[')
        if len(split) == 1:
            path.append((split[0].lower(), None))
        else:
            path.append((split[0].lower(), int(split[1][:-1])))

    return path


def html_to_plaintext(html, keep_paragraphs_only=False, trim_start=None, lowercase=True, merge_whitespaces=True):
    soup = BeautifulSoup(html, 'lxml')

    if keep_paragraphs_only:
        soup_text = keep_paragraphs(soup)
        if trim_start:
            soup_text = trim_text_start_length(soup_text, trim_start)
    else:
        soup_text = soup.get_text()

    if lowercase:
        soup_text = soup_text.lower()

    if merge_whitespaces:
        soup_text = re.sub('\\s+', ' ', soup_text)

    return soup_text
