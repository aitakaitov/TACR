import argparse

import bs4
import pandas as pd
import json

from annotated_dataset.annotation_merge_utils import process_span, get_span_intersections, get_spans_with_intersection, \
    perform_union_merge, perform_intersection_merge, keep_all
from annotated_dataset.html_utils import html_to_plaintext
import re

def get_webpages():
    """
    Loads a list of webpage IDs
    """
    with open('datasets_complete/1_to_0_and_2_removed/webpages.txt', 'r', encoding='utf-8') as f:
        return [w.strip() for w in f.readlines()]


def get_index_for_url(webpages, url):
    """
    Gets the name of the HTML file for a given URL (corresponds to the index in webpages + 1)
    """
    return webpages.index(url) + 1


def main():
    df = pd.read_csv(args['input_file'])
    webpages = get_webpages()

    data_samples = []
    for index, row in df.iterrows():
        # ignore these
        if 'bad/' in row['url'] or 'bad2/' in row['url']:
            continue

        # check agreement
        if row['majority_fraction'] < args['min_fraction']:
            continue

        # calculate number of annotators
        pos_c = 0
        neg_c = 0
        for state in row['states'].split('///'):
            if state == '0':
                pos_c += 1
            else:
                neg_c += 1

        majority_size = max(pos_c, neg_c)
        frac_per_an = row['majority_fraction'] / majority_size
        annotators = majority_size + (1 - row['majority_fraction']) / frac_per_an
        if annotators < args['min_annotators']:
            continue

        classification = 0 if neg_c > pos_c else 1

        with open(f'html/{get_index_for_url(webpages, row["url"])}.html', 'r', encoding='utf-8') as f:
            html = f.read()
            soup = bs4.BeautifulSoup(html)
            text = soup.get_text()
            text = re.sub('\\s+', ' ', text)
            data_samples.append({
                'text': text,
                'label': classification
            })

    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='datasets_complete/1_to_0_and_2_removed/0.5.csv')
    parser.add_argument('--output_file', type=str, required=False, default=None)
    parser.add_argument('--min_fraction', default=0.75, type=float)
    parser.add_argument('--min_annotators', default=3, type=int)
    args = vars(parser.parse_args())

    if args['output_file'] is None:
        # todo change
        args['output_file'] = f'3-75_classification.jsonl'

    # stats = {
    #     'merged_span_counts_positive': 0,
    #     'merged_span_counts_negative': 0,
    #     'merged_span_lengths_positive': [],
    #     'merged_span_lengths_negative': [],
    # }

    main()