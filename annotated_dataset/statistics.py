import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

from annotated_dataset.annotation_merge_utils import process_span, get_span_intersections
from annotated_dataset.html_utils import html_to_plaintext


def get_webpages():
    with open('datasets/1_to_0_and_2_removed/webpages.txt', 'r', encoding='utf-8') as f:
        return [w.strip() for w in f.readlines()]


def get_index_for_url(webpages, url):
    return webpages.index(url) + 1


def get_span_statistics(html, spans, match_leniency, min_span_length, lowercase, whitespace_merge):
    soup_text = html_to_plaintext(html, lowercase=lowercase, merge_whitespaces=whitespace_merge)

    spans_located = 0
    spans_long_enough = 0
    valid_spans = []
    span_lengths_tokens = []
    span_lengths_chars = []
    for span in spans:
        if len(span[1:-1].split()) < min_span_length:
            continue
        result = process_span(span[1:-1], soup_text, lowercase=lowercase, merge_whitespaces=whitespace_merge, leniency=match_leniency)
        if result is not None:
            spans_located += 1
            valid_spans.append(result)
            span_lengths_tokens.append(len(span[1:-1].split()))
            span_lengths_chars.append(len(span[1:-1]))

        spans_long_enough += 1

    simple_matches = 0
    charsim_matches = 0
    spans_with_simple_duplicates = 0
    spans_with_charsim_duplicates = 0
    total_duplicates_simple_match = 0
    total_duplicates_charsim_match = 0
    for span_info in valid_spans:
        if span_info['count_simple'] > 0:
            simple_matches += 1
            if span_info['count_simple'] > 1:
                spans_with_simple_duplicates += 1
                total_duplicates_simple_match += span_info['count_simple'] - 1
        elif span_info['count_sim'] > 0:
            charsim_matches += 1
            if span_info['count_sim'] > 1:
                spans_with_charsim_duplicates += 1
                total_duplicates_charsim_match += span_info['count_sim'] - 1

    intersections = get_span_intersections(valid_spans)
    intersections_count = len(intersections)
    intersections_percents = [i['intersection_percent'] for i in intersections]

    intersected_spans = []
    for i in range(len(valid_spans)):
        for intersection in intersections:
            if i == intersection['span_1'] or i == intersection['span_2']:
                if i not in intersected_spans:
                    intersected_spans.append(i)

    intersectionless_spans = len(valid_spans) - len(intersected_spans)

    return {
        'span_lengths_tokens': span_lengths_tokens,
        'span_lengths_chars': span_lengths_chars,
        'long_enough_spans': spans_long_enough,
        'spans_located': spans_located,
        'spans_matched_simple': simple_matches,
        'spans_matched_charsim': charsim_matches,
        'spans_with_simple_duplicate': spans_with_simple_duplicates,
        'spans_with_charsim_duplicate': spans_with_charsim_duplicates,
        'total_simple_duplicates': total_duplicates_simple_match,
        'total_charsim_duplicates': total_duplicates_charsim_match,
        'intersection_count': intersections_count,
        'intersection_percents': intersections_percents,
        'spans_without_intersection': intersectionless_spans
    }


def get_stats(min_fraction, min_annotators, min_spans, positive_only, match_leniency, min_span_length, lowercase, whitespace_merge):
    print(min_fraction, min_annotators, min_spans, positive_only, match_leniency)
    os.makedirs(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}', exist_ok=True)

    df = pd.read_csv('datasets/1_to_0_and_2_removed/0.7.csv')

    pattern = re.compile(r'[a-z0-9]+\.cz')

    webpages = get_webpages()

    total_docs = 0
    ad_docs = 0

    # domain counts for graphs
    domain_counts_positive = {}
    domain_counts_negative = {}

    # intersection stuff
    intersections_count_total = 0
    intersections_percents = []
    spans_without_intersection = 0


    # total number of spans in the data
    total_spans_available = 0
    # total number of spans long-enough
    total_spans_eligible = 0
    # spans that have been found in the site text
    total_spans_located = 0
    # spans located through exact matching
    total_spans_located_simple_match = 0
    # spans located through charsim matching
    total_spans_located_charsim_match = 0
    # the number of spans that have a duplicate
    total_spans_simple_has_duplicate = 0
    total_spans_charsim_has_duplicate = 0
    # the number of duplicates
    total_duplicates_simple_match = 0
    total_duplicates_charsim_match = 0
    # numbers of spans
    span_counts = []
    # lengths of spans
    span_lengths_tokens = []
    span_lengths_chars = []
    for index, row in df.iterrows():
        if 'bad/' in row['url'] or 'bad2/' in row['url']:
            continue

        if row['majority_fraction'] < min_fraction:
            continue

        frac_per_an = row['majority_fraction'] / row['majority_size']
        annotators = row['majority_size'] + (1 - row['majority_fraction']) / frac_per_an
        if annotators < min_annotators:
            continue

        if row['state'] != 0 and positive_only:
            continue

        if row['span_count'] < min_spans:
            continue

        total_docs += 1
        if row['state'] == 0:
            ad_docs += 1

        url = row['url']
        domain = re.search(pattern, url).group(0)

        if row['state'] == 0:
            if domain not in domain_counts_positive.keys():
                domain_counts_positive[domain] = 1
            else:
                domain_counts_positive[domain] += 1
        else:
            if domain not in domain_counts_negative.keys():
                domain_counts_negative[domain] = 1
            else:
                domain_counts_negative[domain] += 1

        texts = row['text'].split(';')
        total_spans_available += len(texts)

        with open(f'html/{get_index_for_url(webpages, row["url"])}.html', 'r', encoding='utf-8') as f:
            data = get_span_statistics(f.read(), texts, match_leniency, min_span_length, lowercase, whitespace_merge)

        total_spans_eligible += data['long_enough_spans']
        total_spans_located += data['spans_located']
        total_spans_located_simple_match += data['spans_matched_simple']
        total_spans_located_charsim_match += data['spans_matched_charsim']
        total_spans_simple_has_duplicate += data['spans_with_simple_duplicate']
        total_spans_charsim_has_duplicate += data['spans_with_charsim_duplicate']
        total_duplicates_simple_match += data['total_simple_duplicates']
        total_duplicates_charsim_match += data['total_charsim_duplicates']

        span_lengths_chars.extend(data['span_lengths_chars'])
        span_lengths_tokens.extend(data['span_lengths_tokens'])
        span_counts.append(data['spans_located'])

        intersections_count_total += data['intersection_count']
        intersections_percents.extend(data['intersection_percents'])
        spans_without_intersection += data['spans_without_intersection']

    plt.hist(span_counts, bins=10)
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_counts.png')
    plt.xlabel('span counts')
    plt.ylabel('spans')
    plt.close()

    plt.hist(span_lengths_tokens, bins=30)
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_lengths.png')
    plt.xlabel('span lengths in tokens')
    plt.ylabel('spans')
    plt.close()

    for d in domain_counts_positive.keys():
        if d not in domain_counts_negative.keys():
            domain_counts_negative[d] = 0
    for d in domain_counts_negative.keys():
        if d not in domain_counts_positive.keys():
            domain_counts_positive[d] = 0

    domains = domain_counts_negative.keys()

    counts = {
        'Positive': np.array([domain_counts_positive[d] for d in domains]),
        'Negative': np.array([domain_counts_negative[d] for d in domains])
    }

    fig, ax = plt.subplots()
    bottom = np.zeros(len(domain_counts_positive.keys()))

    for boolean, weight_count in counts.items():
        p = ax.bar(domains, weight_count, 0.5, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.ylabel('documents')
    plt.subplots_adjust(bottom=0.3)  # Adjust the value as needed
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/domains.png')
    plt.close()

    return {
        # metadata, class stats
        'positive_only': positive_only,
        'min_frac': '{:.2f}'.format(min_fraction),
        'min_annotators': min_annotators,
        'leniency': match_leniency,
        'min_span_length': min_span_length,
        'lowercase': lowercase,
        'whitespace_merge': whitespace_merge,
        'total_docs': total_docs,
        'ad_docs_perc': ad_docs / total_docs,

        # span statistics
        'total_spans_available': total_spans_available,
        'spans_dropped_for_length_perc': 1 - (total_spans_eligible / total_spans_available),
        'spans_located_perc': total_spans_located / total_spans_eligible,
        'spans_located_simple_match_perc': total_spans_located_simple_match / total_spans_located,
        'spans_located_charsim_match_perc': total_spans_located_charsim_match / total_spans_located,
        'avg_spans_located_per_doc': total_spans_located / total_docs,
        'simple_located_duplicate_perc': total_spans_simple_has_duplicate / total_spans_located_simple_match,
        'simple_located_span_avg_duplicates': total_duplicates_simple_match / total_spans_simple_has_duplicate,
        'charsim_located_duplicate_perc': total_spans_charsim_has_duplicate / total_spans_located_charsim_match if total_spans_located_charsim_match != 0 else 'nan',
        'charsim_located_span_avg_duplicates': total_duplicates_charsim_match / total_spans_charsim_has_duplicate if total_spans_located_charsim_match != 0 else 'nan',

        'avg_spans_per_doc': total_spans_located / total_docs,
        'avg_span_length_tokens': sum(span_lengths_tokens) / total_spans_located,
        'avg_span_length_chars': sum(span_lengths_chars) / total_spans_located,

        # intersection stats
        'intersections_per_span': intersections_count_total / total_spans_located,
        'intersections_per_doc': intersections_count_total / total_docs,
        'average_intersection_size_perc': sum(intersections_percents) / len(intersections_percents),
        'spans_with_intersect_perc': 1 - (spans_without_intersection / total_spans_located),
    }


def main():
    majority_fractions = [2 / 3.0]
    min_annotators = [3]
    min_spans = [3]
    match_leniency = [1000, 2000, 5000]
    positive_only = [False]
    min_span_lengths = [1] #, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    lowercase = [True]
    whitespace_merge = [True]

    first = True

    with open('stats.csv', 'w+', encoding='utf-8') as f:
        for po in positive_only:
            for frac in majority_fractions:
                for anns in min_annotators:
                    for sp in min_spans:
                        for l in match_leniency:
                            for msl in min_span_lengths:
                                for lwc in lowercase:
                                    for wsm in whitespace_merge:
                                        result = get_stats(frac, anns, sp, po, l, msl, lwc, wsm)
                                        if first:
                                            first = False
                                            f.write(';'.join(result.keys()) + '\n')

                                        f.write(';'.join([str(x) for x in result.values()]) + '\n')


if __name__ == '__main__':
    main()
