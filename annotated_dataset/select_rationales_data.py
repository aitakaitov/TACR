import json
import pandas as pd
import argparse
from bs4 import BeautifulSoup

from annotated_dataset.html_utils import process_html_full


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

        data.append({
            'url': row['url'],
            'rationales': rationales,
            'html': html,
            'text': process_html_full(html, args['trim_text'], args['trim_length'])
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