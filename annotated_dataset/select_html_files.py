import json
import os
import argparse
import pandas as pd

from annotated_dataset.html_utils import process_html_full


def load_csv():
    with open(os.path.join('datasets', '1_to_0_and_2_removed', f'{args["fraction"]}.csv'), 'r', encoding='utf-8') as f:
        df = pd.read_csv(f)

    data = []
    for index, row in df.iterrows():
        if args['all_agree']:
            if row['majority_fraction'] < 1:
                print('rejecting due to majority fraction less than 1')
                continue

        frac_per_an = row['majority_fraction'] / row['majority_size']
        annotators = row['majority_size'] + (1 - row['majority_fraction']) / frac_per_an
        if annotators < args['min_annotators']:
            print('rejecting due to annotators')
            continue

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
        sample['text'] = process_html_full(sample['html'], args['trim_text'], args['trim_length'])

    return data_dict


def transform_labels(data_dict):
    for sample in data_dict:
        sample['label'] = 0 if sample['label'] == 3 else 1

    return data_dict


def export(data_dict):
    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for sample in data_dict:
            print(sample['label'])
            f.write(json.dumps(
                {
                    'text': sample['text'],
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


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=False, default='dataset_annotated.jsonl')
    parser.add_argument('--trim_text', default=None, required=False)
    parser.add_argument('--trim_length', default=0, required=False, type=int)
    parser.add_argument('--min_annotators', default=1, type=int, required=False)
    parser.add_argument('--all_agree', default=False, type=parse_bool, required=False)
    args = vars(parser.parse_args())

    main()