import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

from annotated_dataset.annotation_merge_utils import process_span, get_span_intersections, get_spans_with_intersection
from annotated_dataset.html_utils import html_to_plaintext


def get_webpages():
    """
    Loads a list of webpage IDs
    """
    with open('datasets/1_to_0_and_2_removed/webpages.txt', 'r', encoding='utf-8') as f:
        return [w.strip() for w in f.readlines()]


def get_index_for_url(webpages, url):
    """
    Gets the name of the HTML file for a given URL (corresponds to the index in webpages + 1)
    """
    return webpages.index(url) + 1


def get_span_statistics(html, span_data_list, match_strictness, min_span_length, lowercase, whitespace_merge, classes):
    """
    Given the HTML string, a list of spans and hyperparameters, calculate a set of doc-specific statistics
    @param html: HTML string
    @param span_data_list: list of span data - can be changed
    @param match_strictness: how strict the inexact matching is (value of ~10 000 will disable inexact matches)
    @param min_span_length: minimum length of span text in tokens (span is split by whitespace, set to 0 to disable)
    @param lowercase: lowercase the span texts and the extracted HTML text
    @param whitespace_merge: merge sequences of whitespaces into a single space
    @return: statistics
    """
    soup_text = html_to_plaintext(html, lowercase=lowercase, merge_whitespaces=whitespace_merge)

    # how many spans passed the length check
    spans_long_enough = 0
    # how many spans we have identified as present
    spans_located = 0
    # list of long enough and found spans
    valid_spans = []
    # lengths of spans
    span_lengths_tokens = []
    span_lengths_chars = []
    for spans_data in span_data_list:
        spans = []
        for span in spans_data['spans']:
            # check length
            if len(span[1:-1].split()) < min_span_length:
                continue
            # try to find the span
            result = process_span(span[1:-1], soup_text, lowercase=lowercase, merge_whitespaces=whitespace_merge, strictness=match_strictness)
            if result is not None:
                # if span is found, record stats
                spans_located += 1
                spans.append(result)
                span_lengths_tokens.append(len(span[1:-1].split()))
                span_lengths_chars.append(len(span[1:-1]))

            spans_long_enough += 1

        valid_spans.append({
            'annotator_id': spans_data['annotator_id'],
            'decision': spans_data['decision'],
            'spans': spans
        })

    # exact match count
    simple_matches = 0
    # inexact match count
    charsim_matches = 0
    # count of exact matches which had duplicates
    spans_with_simple_duplicates = 0
    # count of inexact matches that had duplicates
    spans_with_charsim_duplicates = 0
    # count of duplicates in exact matches
    total_duplicates_simple_match = 0
    # count of duplicates in inexact matches
    total_duplicates_charsim_match = 0
    for annotator in valid_spans:
        for span_info in annotator['spans']:
            # check if the match was exact
            if span_info['count_simple'] > 0:
                simple_matches += 1
                # check for duplicates
                if span_info['count_simple'] > 1:
                    spans_with_simple_duplicates += 1
                    total_duplicates_simple_match += span_info['count_simple'] - 1
            # otherwise check if it was inexact
            elif span_info['count_sim'] > 0:
                charsim_matches += 1
                # check for duplicates
                if span_info['count_sim'] > 1:
                    spans_with_charsim_duplicates += 1
                    total_duplicates_charsim_match += span_info['count_sim'] - 1

    # get intersection stats
    if classes != 'both':
        if classes == 'ads':
            spans_to_process = []
            for a in valid_spans:
                if a['decision'] == 'positive':
                    spans_to_process.extend(a['spans'])
            intersections = get_span_intersections(spans_to_process)
        else:
            spans_to_process = []
            for a in valid_spans:
                if a['decision'] == 'negative':
                    spans_to_process.extend(a['spans'])
            intersections = get_span_intersections(spans_to_process)

        intersections_count = len(intersections)
        intersections_percents = [i['intersection_percent'] for i in intersections]
        intersected_spans = get_spans_with_intersection(intersections, spans_to_process)
        intersectionless_spans = len(spans_to_process) - len(intersected_spans)

    else:
        spans_to_process_positive = []
        spans_to_process_negative = []
        for a in valid_spans:
            if a['decision'] == 'negative':
                spans_to_process_negative.extend(a['spans'])
            else:
                spans_to_process_positive.extend(a['spans'])

        intersections_positive = get_span_intersections(spans_to_process_positive)
        intersections_negative = get_span_intersections(spans_to_process_negative)
        intersected_spans_positive = get_spans_with_intersection(intersections_positive, spans_to_process_positive)
        intersected_spans_negative = get_spans_with_intersection(intersections_negative, spans_to_process_negative)

        intersected_spans = intersected_spans_positive
        intersected_spans.extend(intersected_spans_negative)
        intersections_count = len(intersections_positive) + len(intersections_negative)

        intersections_percents = [i['intersection_percent'] for i in intersections_positive]
        intersections_percents.extend(i['intersection_percent'] for i in intersections_negative)

        intersectionless_spans = len(spans_to_process_positive) + len(spans_to_process_negative) - len(intersected_spans)

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


def get_stats(min_fraction, min_annotators, min_spans, positive_docs_only, match_leniency, min_span_length, lowercase, whitespace_merge, classes):
    os.makedirs(f'histograms/po-{positive_docs_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}', exist_ok=True)
    df = pd.read_csv('datasets_complete/1_to_0_and_2_removed/0.7.csv')
    pattern = re.compile(r'[a-z0-9]+\.cz')
    webpages = get_webpages()

    # counts of kinds of documents
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
        # ignore these
        if 'bad/' in row['url'] or 'bad2/' in row['url']:
            continue

        # check agreement
        if row['majority_fraction'] < min_fraction:
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
        if annotators < min_annotators:
            continue

        classification = [int(d) for d in row['states'].split('///')]
        classification = 'negative' if sum(classification) / len(classification) > 1.5 else 'positive'

        # if positive only, ignore negative
        if classification == 'negative' and positive_docs_only:
            continue

        # check number of spans
        if sum([int(c) for c in row['span_counts'].split('///')]) < min_spans:
            continue

        # record docs
        total_docs += 1
        if classification == 'positive':
            ad_docs += 1

        # extract domain for stats
        url = row['url']
        domain = re.search(pattern, url).group(0)

        if classification == 'positive':
            if domain not in domain_counts_positive.keys():
                domain_counts_positive[domain] = 1
            else:
                domain_counts_positive[domain] += 1
        else:
            if domain not in domain_counts_negative.keys():
                domain_counts_negative[domain] = 1
            else:
                domain_counts_negative[domain] += 1

        # process spans
        annotator_ids = row['emails'].split('///')
        annotator_span_counts = [int(c) for c in row['span_counts'].split('///')]
        annotator_decisions = ['positive' if d == '0' else 'negative' for d in row['states'].split('///')]
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
        if classes == 'both':
            indices_to_remove = []
        # 'ads' means we want spans only from positive annotators
        elif classes == 'ads':
            indices_to_remove = []
            for i in range(len(annotator_decisions)):
                if annotator_decisions[i] == 'negative':
                    indices_to_remove.append(i)
        # 'non_ads' means we want spans only from negative annotators
        elif classes == 'non_ads':
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

        total_spans_available += sum([len(s) for s in annotator_spans])

        with open(f'html/{get_index_for_url(webpages, row["url"])}.html', 'r', encoding='utf-8') as f:
            data = get_span_statistics(f.read(), span_data, match_leniency, min_span_length, lowercase, whitespace_merge, classes)

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

    # plt.hist(span_counts, bins=10)
    # plt.savefig(f'histograms/po-{positive_docs_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_counts.png')
    # plt.xlabel('span counts')
    # plt.ylabel('spans')
    # plt.close()
    #
    # plt.hist(span_lengths_tokens, bins=30)
    # plt.savefig(f'histograms/po-{positive_docs_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/span_lengths.png')
    # plt.xlabel('span lengths in tokens')
    # plt.ylabel('spans')
    # plt.close()
    #
    # for d in domain_counts_positive.keys():
    #     if d not in domain_counts_negative.keys():
    #         domain_counts_negative[d] = 0
    # for d in domain_counts_negative.keys():
    #     if d not in domain_counts_positive.keys():
    #         domain_counts_positive[d] = 0
    #
    # domains = domain_counts_negative.keys()
    #
    # counts = {
    #     'Positive': np.array([domain_counts_positive[d] for d in domains]),
    #     'Negative': np.array([domain_counts_negative[d] for d in domains])
    # }
    #
    # fig, ax = plt.subplots()
    # bottom = np.zeros(len(domain_counts_positive.keys()))
    #
    # for boolean, weight_count in counts.items():
    #     p = ax.bar(domains, weight_count, 0.5, label=boolean, bottom=bottom)
    #     bottom += weight_count
    #
    # ax.legend(loc="upper right")
    # plt.xticks(rotation=90)
    # plt.ylabel('documents')
    # plt.subplots_adjust(bottom=0.3)  # Adjust the value as needed
    # plt.savefig(f'histograms/po-{positive_docs_only}_f{min_fraction:.2f}_a{min_annotators}_s{min_spans}/domains.png')
    # plt.close()

    return {
        # metadata, class stats
        'span_classes': classes,
        'positive_docs_only': positive_docs_only,
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
        'charsim_located_span_avg_duplicates': total_duplicates_charsim_match / total_spans_charsim_has_duplicate if total_spans_charsim_has_duplicate != 0 else 'nan',

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
    match_leniency = [100, 250, 500, 1000, 2000, 5000]
    positive_only = [False]
    min_span_lengths = [1] #, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    lowercase = [True]
    whitespace_merge = [True]
    classes = ['ads', 'non_ads', 'both']

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
                                        for c in classes:
                                            result = get_stats(frac, anns, sp, po, l, msl, lwc, wsm, c)
                                            if first:
                                                first = False
                                                f.write(';'.join(result.keys()) + '\n')

                                            f.write(';'.join([str(x) for x in result.values()]) + '\n')


if __name__ == '__main__':
    main()
