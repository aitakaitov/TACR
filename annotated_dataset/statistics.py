import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup


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


def get_node_by_path(soup, path):
    current_node = soup
    for node, index in path[1:]:
        if node == '#text':
            break
        current_node = current_node.find_all_next('')

    return current_node

def get_overlap_between_spans(html_file, texts, start_paths, start_offsets, end_paths, end_offsets, extend_by_tokens=0):
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html)

    for sp, so, ep, eo in zip(start_paths, start_offsets, end_paths, end_offsets):
        start_path_nodes = parse_xpath(sp[1:-1])
        end_path_nodes = parse_xpath(ep[1:-1])
        start_node = get_node_by_path(soup, start_path_nodes)
        end_node = get_node_by_path(soup, end_path_nodes)

        pass


def get_stats(min_fraction, min_annotators, min_spans, positive_only):
    print(min_fraction, min_annotators, min_spans, positive_only)
    df = pd.read_csv('datasets/1_to_0_and_2_removed/0.7.csv')

    os.makedirs(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}', exist_ok=True)

    expr = re.compile(r'[a-z0-9]+\.cz')

    total_docs = 0
    ad_docs = 0
    span_counts = []
    span_lengths = []
    domain_counts_positive = {}
    domain_counts_negative = {}
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
        domain = re.search(expr, url).group(0)

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

        get_overlap_between_spans(f'html/{index}.html', texts, start_paths, start_offsets, end_paths, end_offsets)

    plt.hist(span_counts, bins=10)
    plt.savefig(f'histograms/po-{positive_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_counts.png')
    plt.xlabel('span counts')
    plt.ylabel('spans')
    plt.close()

    plt.hist(span_lengths, bins=10)
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

    return (positive_only, '{:.2f}'.format(min_fraction), min_annotators, min_spans, total_docs, ad_docs, ad_docs / total_docs,
            np.mean(span_counts), np.percentile(span_counts, 25), np.percentile(span_counts, 50), np.percentile(span_counts, 75),
            np.mean(span_lengths), np.percentile(span_lengths, 25), np.percentile(span_lengths, 50), np.percentile(span_lengths, 75))


def main():
    majority_fractions = [2 / 3.0, 1.0]
    min_annotators = [3]
    min_spans = [3, 4, 5, 6]
    positive_only = [True, False]

    with open('stats.csv', 'w+', encoding='utf-8') as f:
        f.write('positive_only;min_fraction;min_annotators;min_spans;total_docs;ad_docs;ad_docs_percent;'
                'avg_spans;p25_spans;p50_spans;p75_spans;'
                'avg_span_len;p25_span_len;p50_span_len;p75_span_len\n')
        for po in positive_only:
            for frac in majority_fractions:
                for anns in min_annotators:
                    for sp in min_spans:
                        f.write(
                            ';'.join(str(x) for x in get_stats(frac, anns, sp, po)) + '\n'
                        )


if __name__ == '__main__':
    main()