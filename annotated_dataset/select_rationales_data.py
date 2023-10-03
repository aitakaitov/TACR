import json
import pandas as pd
import argparse
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


def trim_text_start_length(text):
    MIN_TOKENS = args['trim_length']
    tag_texts = text.split('\n')
    start = -1
    for i, tag in enumerate(tag_texts):
        if len(tag.split()) > MIN_TOKENS:
            start = i
            break

    new_text = ''
    for i in range(start, len(tag_texts)):
        new_text += tag_texts[i]

    return new_text


def trim_text(text):
    if args['trim_text'] is None:
        return text
    elif args['trim_text'] == 'start_length':
        return trim_text_start_length(text)
    else:
        print(f'{args["trim_text"]} not valid')
        exit(-1)


def main():
    df = pd.read_csv(args['input_file'])

    data = []
    for index, row in df.iterrows():
        if row['majority_fraction'] < args['majority_fraction']:
            continue

        frac_per_an = row['majority_fraction'] / row['majority_size']
        annotators = row['majority_size'] + (1 - row['majority_fraction']) / frac_per_an
        if annotators < args['min_annotators']:
            continue

        if row['state'] != 0 and args['positive_only']:
            continue

        if row['span_count'] < args['min_spans']:
            continue

        rationales = []
        start_paths = row['start_path'].split(';')
        start_offsets = row['start_offset'].split(';')
        end_paths = row['end_path'].split(';')
        end_offsets = row['end_offset'].split(';')
        texts = row['text'].split(';')

        for sp, so, ep, eo, tx in zip(start_paths, start_offsets, end_paths, end_offsets, texts):
            rationales.append({
                'start_path': sp[1:-1],
                'start_offset': int(so[1:-1]),
                'end_path': ep[1:-1],
                'end_offset': int(eo[1:-1]),
                'text': tx[1:-1]
            })

        with open(f'html/{int(index) + 1}.html', 'r', encoding='utf-8') as f:
            html = f.read()
            soup = BeautifulSoup(html)
            soup = filter_html(soup)
            extracted = keep_paragraphs(soup)
            extracted = trim_text(extracted)

        data.append({
            'url': row['url'],
            'rationales': rationales,
            'html': html,
            'text': extracted
        })

    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='datasets/1_to_0_and_2_removed/0.7.csv')
    parser.add_argument('--output_file', type=str, required=False, default=None)
    parser.add_argument('--majority_fraction', default=1.0, type=float)
    parser.add_argument('--min_annotators', default=3, type=int)
    parser.add_argument('--positive_only', default=True, type=parse_bool)
    parser.add_argument('--min_spans', default=2, type=int)
    parser.add_argument('--trim_text', default='start_length')
    parser.add_argument('--trim_length', default=15)
    args = vars(parser.parse_args())

    if args['output_file'] is None:
        args['output_file'] = (f'rationales_positive-{args["positive_only"]}_maj-{args["majority_fraction"]}_anno-'
                               f'{args["min_annotators"]}_spans-{args["min_spans"]}.jsonl')

    main()