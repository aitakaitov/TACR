import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup

from annotated_dataset.html_utils import html_to_plaintext
from annotated_dataset.similarity_utils import character_count_similarity_index


def get_webpages():
    with open('datasets/1_to_0_and_2_removed/webpages.txt', 'r', encoding='utf-8') as f:
        return [w.strip() for w in f.readlines()]


def get_index_for_url(webpages, url):
    return webpages.index(url) + 1


def get_span_statistics(html, texts, start_paths, start_offsets, end_paths, end_offsets, match_leniency):
    soup_text = html_to_plaintext(html, lowercase=True, merge_whitespaces=True)

    spans_located = 0
    spans_total = 0
    for text in texts:
        text = re.sub('\\s+', ' ', text).lower()[1:-1]
        if text in soup_text:
            # try simple match
            spans_located += 1
        else:
            # if simple match fails, try to match a set of characters allowing for <leniency> different characters
            if character_count_similarity_index(text, soup_text, match_leniency):
                spans_located += 1

        spans_total += 1

    return spans_total, spans_located


def get_stats(min_fraction, min_annotators, min_spans, positive_only, match_leniency):
    print(min_fraction, min_annotators, min_spans, positive_only, match_leniency)
    os.makedirs(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}', exist_ok=True)

    df = pd.read_csv('datasets/1_to_0_and_2_removed/0.7.csv')

    pattern = re.compile(r'[a-z0-9]+\.cz')

    webpages = get_webpages()

    total_docs = 0
    ad_docs = 0
    span_counts = []
    span_lengths = []
    domain_counts_positive = {}
    domain_counts_negative = {}
    spans_total = 0
    spans_located = 0
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

        span_counts.append(len(row['start_path'].split(';')))

        for text in row['text'].split(';'):
            text = text[1:-1]
            split = text.split()
            span_lengths.append(len(split))

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

        start_paths = row['start_path'].split(';')
        start_offsets = row['start_offset'].split(';')
        end_paths = row['end_path'].split(';')
        end_offsets = row['end_offset'].split(';')
        texts = row['text'].split(';')

        with open(f'html/{get_index_for_url(webpages, row["url"])}.html', 'r', encoding='utf-8') as f:
            total, located = get_span_statistics(f.read(), texts, start_paths, start_offsets, end_paths, end_offsets, match_leniency)

        spans_total += total
        spans_located += located

    plt.hist(span_counts, bins=10)
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_counts.png')
    plt.xlabel('span counts')
    plt.ylabel('spans')
    plt.close()

    plt.hist(span_lengths, bins=30)
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_lengths.png')
    plt.xlabel('span lengths')
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

    return (positive_only, '{:.2f}'.format(min_fraction), min_annotators, min_spans, match_leniency, total_docs, ad_docs, ad_docs / total_docs,
            spans_located / float(spans_total), np.mean(span_counts), np.percentile(span_counts, 25), np.percentile(span_counts, 50), np.percentile(span_counts, 75),
            np.mean(span_lengths), np.percentile(span_lengths, 25), np.percentile(span_lengths, 50), np.percentile(span_lengths, 75))


def main():
    #majority_fractions = [2 / 3.0, 1.0]
    majority_fractions = [2 / 3.0]
    min_annotators = [3]
    min_spans = [3]
    #min_spans = [3, 4, 5, 6]
    match_leniency = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    #positive_only = [Ttue, False]
    positive_only = [False]

    with open('stats.csv', 'w+', encoding='utf-8') as f:
        f.write('positive_only;min_fraction;min_annotators;min_spans;leniency;total_docs;ad_docs;ad_docs_percent;'
                'spans_located_perc;avg_spans;p25_spans;p50_spans;p75_spans;'
                'avg_span_len;p25_span_len;p50_span_len;p75_span_len\n')
        for po in positive_only:
            for frac in majority_fractions:
                for anns in min_annotators:
                    for sp in min_spans:
                        for l in match_leniency:
                            f.write(
                                ';'.join(str(x) for x in get_stats(frac, anns, sp, po, l)) + '\n'
                            )


if __name__ == '__main__':
    main()
