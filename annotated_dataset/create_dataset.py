import argparse

import pandas as pd
import json

from annotated_dataset.annotation_merge_utils import process_span, get_span_intersections, get_spans_with_intersection, \
    perform_union_merge, perform_intersection_merge, keep_all
from annotated_dataset.html_utils import html_to_plaintext


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


def perform_merge(annotations):
    if args['merge_annotations'] == 'union':
        result = perform_union_merge(annotations)
    elif args['merge_annotations'] == 'intersection':
        result = perform_intersection_merge(annotations)
    elif args['merge_annotations'] == 'keep_all':
        result = keep_all(annotations)

    return result


def process_document(html, annotation_list, classification):
    # find the spans
    soup_text = html_to_plaintext(html, lowercase=args['lowercase'], merge_whitespaces=args['merge_whitespaces'], keep_paragraphs_only=args['keep_paragraphs_only'], trim_start=args['start_trim'])
    annotator_data = []
    for annotation in annotation_list:
        spans = []
        for span in annotation['spans']:
            # check length
            span_length = len(span[1:-1].split())
            if span_length < args['min_span_length'] or span_length > args['max_span_length']:
                continue

            # try to find the span
            result = process_span(span[1:-1], soup_text, lowercase=args['lowercase'],
                                  merge_whitespaces=args['merge_whitespaces'], strictness=args['strictness'],
                                  report_multiples=False)
            if result is not None:
                # if span is found
                spans.append(result)

        annotator_data.append({
            'annotator_id': annotation['annotator_id'],
            'decision': annotation['decision'],
            'spans': spans
        })

    # get intersections
    positive_annotators = []
    [positive_annotators.append(ad) if ad['decision'] == 1 else None for ad in annotator_data]

    negative_annotators = []
    [negative_annotators.append(ad) if ad['decision'] == 0 else None for ad in annotator_data]

    positive_merged = perform_merge(positive_annotators)
    negative_merged = perform_merge(negative_annotators)

    result = {
        'label': classification,
        'text': soup_text
    }

    if args['span_classes'] == 'both' or args['span_classes'] == 'ads':
        result['positive_spans'] = positive_merged
        # for span in positive_merged:
        #     stats['merged_span_counts_positive'] += 1
        #     stats['merged_span_lengths_positive'].append(span['length'])

    if args['span_classes'] == 'both' or args['span_classes'] == 'non_ads':
        result['negative_spans'] = negative_merged
        # for span in positive_merged:
        #     stats['merged_span_counts_negative'] += 1
        #     stats['merged_span_lengths_negative'].append(span['length'])

    return result


def main():
    df = pd.read_csv(args['input_file'])
    webpages = get_webpages()

    no_intersections = 0
    fraction = 0
    annotators_e = 0
    spans_e = 0

    data_samples = []
    for index, row in df.iterrows():
        # ignore these
        if 'bad/' in row['url'] or 'bad2/' in row['url']:
            continue

        # check agreement
        if row['majority_fraction'] < args['min_fraction']:
            fraction += 1
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
            annotators_e += 1
            continue

        classification = 0 if neg_c > pos_c else 1

        # if positive only, ignore negative
        if classification == 0 and args['positive_only']:
            continue

        # check number of spans
        if sum([int(c) for c in row['span_counts'].split('///')]) < args['min_spans']:
            spans_e += 1
            continue

        # process spans
        annotator_ids = row['emails'].split('///')
        annotator_span_counts = [int(c) for c in row['span_counts'].split('///')]
        annotator_decisions = [1 if d == '0' else 0 for d in row['states'].split('///')]
        annotator_spans = [t.split(';;;') for t in row['texts'].split('///')]
        # check if someone didn't mark any spans
        indices_to_remove = []
        for i in range(len(annotator_ids)):
            if annotator_span_counts[i] == 0:
                indices_to_remove.append(i)

        # remove them
        indices_to_remove.reverse()
        for i in indices_to_remove:
            del annotator_ids[i]
            del annotator_span_counts[i]
            del annotator_decisions[i]
            del annotator_spans[i]

        # now check which spans we want
        # 'both' means we want spans from both positive and negative annotators
        if args['span_classes'] == 'both':
            indices_to_remove = []
        # 'ads' means we want spans only from positive annotators
        elif args['span_classes'] == 'ads':
            indices_to_remove = []
            for i in range(len(annotator_decisions)):
                if annotator_decisions[i] == 'negative':
                    indices_to_remove.append(i)
        # 'non_ads' means we want spans only from negative annotators
        elif args['span_classes'] == 'non_ads':
            indices_to_remove = []
            for i in range(len(annotator_decisions)):
                if annotator_decisions[i] == 'positive':
                    indices_to_remove.append(i)

        indices_to_remove.reverse()
        # remove the selected spans
        for i in indices_to_remove:
            del annotator_ids[i]
            del annotator_span_counts[i]
            del annotator_decisions[i]
            del annotator_spans[i]

        span_data = []
        for annotator_id, annotator_decision, annotator_span in zip(annotator_ids, annotator_decisions, annotator_spans):
            span_data.append({
                'annotator_id': annotator_id,
                'decision': annotator_decision,
                'spans': annotator_span
            })

        with open(f'html/{get_index_for_url(webpages, row["url"])}.html', 'r', encoding='utf-8') as f:
            data = process_document(f.read(), span_data, classification)

        if len(data['negative_spans']) != 0 or len(data['positive_spans']) != 0:
            data_samples.append(data)
        else:
            no_intersections += 1

    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')

    print(f'not enough spans: {spans_e}')
    print(f'no intersections: {no_intersections}')
    print(f'fraction too low: {fraction}')
    print(f'not enough annotators: {annotators_e}')


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='datasets_complete/1_to_0_and_2_removed/0.5.csv')
    parser.add_argument('--output_file', type=str, required=False, default=None)
    parser.add_argument('--min_fraction', default=0.6, type=float)
    parser.add_argument('--min_annotators', default=2, type=int)
    parser.add_argument('--positive_only', default=False, type=parse_bool)
    parser.add_argument('--min_spans', default=2, type=int)
    parser.add_argument('--min_span_length', default=2, type=int)
    parser.add_argument('--max_span_length', default=1500, type=int)
    parser.add_argument('--lowercase', default=True, type=parse_bool)
    parser.add_argument('--merge_whitespaces', default=True, type=parse_bool)
    parser.add_argument('--strictness', default=250, type=int)
    parser.add_argument('--span_classes', default='both', type=str)   # ads, non_ads, both
    parser.add_argument('--merge_annotations', default='keep_all', type=str)  # intersection, union, keep_all
    parser.add_argument('--keep_paragraphs_only', default=False, type=parse_bool)
    parser.add_argument('--start_trim', default=None, type=int)
    args = vars(parser.parse_args())

    if args['output_file'] is None:
        # todo change
        args['output_file'] = f'aaa.jsonl'

    # stats = {
    #     'merged_span_counts_positive': 0,
    #     'merged_span_counts_negative': 0,
    #     'merged_span_lengths_positive': [],
    #     'merged_span_lengths_negative': [],
    # }

    main()

    # print(f'Merged spans positive count: {stats["merged_span_counts_positive"]}')
    # print(f'Merged spans negative count: {stats["merged_span_counts_negative"]}')
    # print(f'Merged spans positive average length (chars): {sum(stats["merged_span_lengths_positive"]) / len(stats["merged_span_lengths_positive"])}')
    # print(f'Merged spans negative average length (chars): {sum(stats["merged_span_lengths_negative"]) / len(stats["merged_span_lengths_negative"])}')
    # print(f'Merged spans positive max length (chars): {max(stats["merged_span_lengths_positive"])}')
    # print(f'Merged spans negative max length (chars): {max(stats["merged_span_lengths_negative"])}')
    # print(f'Merged spans positive min length (chars): {min(stats["merged_span_lengths_positive"])}')
    # print(f'Merged spans negative min length (chars): {min(stats["merged_span_lengths_negative"])}')