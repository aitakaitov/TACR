import json
import argparse
import pandas as pd
import re

from annotated_dataset.html_utils import html_to_plaintext
from similarity_utils import character_count_similarity_index


def validate_span(span_text, plaintext_doc):
    if args['lowercase']:
        span_text = span_text.lower()

    if args['merge_whitespaces']:
        span_text = re.sub('\\s+', ' ', span_text)

    span_length = len(span_text)

    try:
        index = plaintext_doc.index(span_text)
    except ValueError:
        index = character_count_similarity_index(span_text, plaintext_doc, leniency=args['leniency'])
        if not index:
            return None

    return {'start_index': index, 'length': span_length}


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
        texts = [t[1:-1] for t in row['text'].split(';')]

        with open(f'html/{int(index) + 1}.html', 'r', encoding='utf-8') as f:
            html = f.read()

        plaintext = html_to_plaintext(html, args['lowercase'], args['merge_whitespaces'])

        for text in texts:
            span_data = validate_span(text, plaintext)
            if not span_data:
                continue
            else:
                rationales.append(span_data)

        data.append({
            'url': row['url'],
            'rationales': rationales,
            'html': html,
            'text': plaintext,
            'config': {
                'lowercase': args['lowercase'],
                'merge_whitespaces': args['merge_whitespaces'],
                'match_leniency': args['leniency']
            }
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
    parser.add_argument('--positive_only', default=False, type=parse_bool)
    parser.add_argument('--min_spans', default=3, type=int)
    # parser.add_argument('--trim_text', default='start_length')
    # parser.add_argument('--trim_length', default=15)
    parser.add_argument('--lowercase', default=True, type=parse_bool)
    parser.add_argument('--merge_whitespaces', default=True, type=parse_bool)
    parser.add_argument('--leniency', default=8, type=int)
    args = vars(parser.parse_args())

    if args['output_file'] is None:
        args['output_file'] = (f'rationales_positive-{args["positive_only"]}_maj-{args["majority_fraction"]}_anno-'
                               f'{args["min_annotators"]}_spans-{args["min_spans"]}.jsonl')

    main()