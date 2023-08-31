import json
import os
import re
from bs4 import BeautifulSoup, NavigableString
import argparse
import pandas as pd

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


def load_csv():
    with open(os.path.join('datasets', '1_to_0_and_2_removed', f'{args["fraction"]}.csv'), 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)

    data = []
    for index, row in df.iterrows():
        url = row['url']
        label = row['state']
        data.append({
            'url': url,
            'label': label
        })

    return data


def load_files(data_dict):
    webpages = []
    with open(os.path.join('datasets', '1_to_0_and_2_removed', 'webpages.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        webpages.append(line.strip())

    for sample in data_dict:
        url = sample['url']
        index = webpages.index(url) + 1

        with open(os.path.join('html', f'{index}.html'), 'r', encoding='utf-8') as f:
            html = f.read()

        sample['html'] = html

    return data_dict


def process_html(data_dict):
    for sample in data_dict:
        soup = BeautifulSoup(sample['html'])
        soup = filter_html(soup)
        sample['text'] = keep_paragraphs(soup)

    return data_dict


def transform_labels(data_dict):
    for sample in data_dict:
        sample['label'] = 0 if sample['label'] == 3 else 1

    return data_dict


def export(data_dict):
    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for sample in data_dict:
            f.write(json.dumps(
                {
                    'text': trim_text(sample['text']),
                    'label': sample['label'],
                    'file': sample['url']
                }
            ) + '\n')


def main():
    # read the csv file
    data_dict = load_csv()
    # get the html files
    data_dict = load_files(data_dict)
    # process the html
    data_dict = process_html(data_dict)
    # transform labels
    data_dict = transform_labels(data_dict)
    # save to jsonl as a dataset
    export(data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False, default='dataset_annotated.jsonl')
    parser.add_argument('--trim_text', default=None, required=False)
    parser.add_argument('--trim_length', default=0, required=False, type=int)
    args = vars(parser.parse_args())

    main()