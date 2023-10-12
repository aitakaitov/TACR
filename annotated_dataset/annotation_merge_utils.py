import re

from similarity_utils import character_count_similarity_index


def merge_intervals(intervals):
    if len(intervals) == 0:
        return []

    # check if any intersections overlap and merge them if they do
    intervals.sort(key=lambda x: x['start_index'])
    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        current_interval = intervals[i]
        last_merged_interval = merged_intervals[-1]

        if current_interval['start_index'] <= last_merged_interval['start_index'] + last_merged_interval['length']:
            last_merged_interval['length'] = max(last_merged_interval['start_index'] + last_merged_interval['length'],
                                                 current_interval['start_index'] + current_interval['length']) - last_merged_interval['start_index']
        else:
            merged_intervals.append(current_interval)

    return merged_intervals


def perform_union_merge(annotations):
    if len(annotations) < 2:
        return []

    unions = []
    # get intersections between spans from different annotators
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            spans_i = annotations[i]['spans']
            spans_j = annotations[j]['spans']

            for s_i in range(len(spans_i)):
                for s_j in range(s_i, len(spans_j)):
                    s_i_end = spans_i[s_i]['start_index'] + spans_i[s_i]['length']
                    s_j_end = spans_j[s_j]['start_index'] + spans_j[s_j]['length']

                    if s_i_end < spans_j[s_j]['start_index'] or s_j_end < spans_i[s_i]['start_index']:
                        continue

                    union_start = min(spans_i[s_i]['start_index'], spans_j[s_j]['start_index'])
                    union_end = max(s_i_end, s_j_end)
                    unions.append({
                        'start_index': union_start,
                        'length': union_end - union_start
                    })

    return merge_intervals(unions)


def keep_all(annotations):
    if len(annotations) < 1:
        return []

    unions = []
    # get intersections between spans from different annotators
    for i in range(len(annotations)):
        for j in range(i, len(annotations)):
            spans_i = annotations[i]['spans']
            spans_j = annotations[j]['spans']

            for s_i in range(len(spans_i)):
                for s_j in range(s_i, len(spans_j)):
                    s_i_end = spans_i[s_i]['start_index'] + spans_i[s_i]['length']
                    s_j_end = spans_j[s_j]['start_index'] + spans_j[s_j]['length']

                    if s_i_end < spans_j[s_j]['start_index'] or s_j_end < spans_i[s_i]['start_index']:
                        continue

                    union_start = min(spans_i[s_i]['start_index'], spans_j[s_j]['start_index'])
                    union_end = max(s_i_end, s_j_end)
                    unions.append({
                        'start_index': union_start,
                        'length': union_end - union_start
                    })

    return merge_intervals(unions)


def perform_intersection_merge(annotations):
    if len(annotations) < 2:
        return []

    intersections = []
    # get intersections between spans from different annotators
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            spans_i = annotations[i]['spans']
            spans_j = annotations[j]['spans']

            for s_i in range(len(spans_i)):
                for s_j in range(s_i, len(spans_j)):
                    s_i_end = spans_i[s_i]['start_index'] + spans_i[s_i]['length']
                    s_j_end = spans_j[s_j]['start_index'] + spans_j[s_j]['length']

                    if s_i_end < spans_j[s_j]['start_index'] or s_j_end < spans_i[s_i]['start_index']:
                        continue

                    intersection_start = max(spans_i[s_i]['start_index'], spans_j[s_j]['start_index'])
                    intersection_end = min(s_i_end, s_j_end)
                    intersections.append({
                        'start_index': intersection_start,
                        'length': intersection_end - intersection_start
                    })

    return merge_intervals(intersections)


def get_spans_with_intersection(intersections, spans):
    intersected_spans = []
    for i in range(len(spans)):
        for intersection in intersections:
            if i == intersection['span_1'] or i == intersection['span_2']:
                if i not in intersected_spans:
                    intersected_spans.append(i)

    return intersected_spans


def process_span(span_text, plaintext_doc, lowercase=True, merge_whitespaces=True, strictness=8, report_multiples=True):
    """
    Finds a span in target text
    @param span_text: what to find
    @param plaintext_doc: where to find it
    @param lowercase: lowercase the span
    @param merge_whitespaces: merge whitespaces of the span into a single space
    @param strictness: how strict the inexact matching is
    @param report_multiples: whether to add statistics on multiple matches
    @return: None when no match, otherwise a dict
    """
    if lowercase:
        span_text = span_text.lower()

    if merge_whitespaces:
        span_text = re.sub('\\s+', ' ', span_text)

    span_length = len(span_text)
    # number of simple matches
    count_simple = 0
    # number of inexact matches
    count_sim = 0
    try:
        # try an exact match
        index = plaintext_doc.index(span_text)
        count_simple = 1
        if report_multiples:
            # find any other matches
            temp_index = index
            while True:
                try:
                    temp_index = plaintext_doc.index(span_text, temp_index + 1)
                    count_simple += 1
                except ValueError:
                    break
    except ValueError:
        # if an exact match does not work, use inexact match
        index = character_count_similarity_index(span_text, plaintext_doc, leniency=strictness)
        count_sim = 1
        if not index:
            # if None is returned, no match was found
            return None
        if report_multiples:
            # if we want to know about multiplicities, run it again
            index, count_sim = character_count_similarity_index(span_text, plaintext_doc, leniency=strictness, check_multiples=True)

    if index == 0:
        print('starts at zero')

    return {
        'start_index': index,
        'length': span_length
    } if not report_multiples else {
        'start_index': index,
        'length': span_length,
        'count_simple': count_simple,
        'count_sim': count_sim
    }


def get_intersection_size(s1_start, s1_len, s2_start, s2_len):
    s1_end = s1_start + s1_len
    s2_end = s2_start + s2_len

    if s1_end < s2_start or s2_end < s1_start:
        return 0

    else:
        intersection_start = max(s1_start, s2_start)
        intersection_end = min(s1_end, s2_end)
        return intersection_end - intersection_start


def get_span_intersections(span_data_list):
    intersections = []
    for i in range(len(span_data_list)):
        for j in range(i + 1, len(span_data_list)):
            span1_end = span_data_list[i]['start_index'] + span_data_list[i]['length']
            span2_end = span_data_list[j]['start_index'] + span_data_list[j]['length']

            if span1_end < span_data_list[j]['start_index'] or span2_end < span_data_list[i]['start_index']:
                continue
            else:
                intersection_start = max(span_data_list[i]['start_index'], span_data_list[j]['start_index'])
                intersection_end = min(span1_end, span2_end)
                intersection_length = intersection_end - intersection_start
                intersections.append({
                    'span_1': i,
                    'span_2': j,
                    'intersection_start': intersection_start,
                    'intersection_end': intersection_end,
                    'intersection_length': intersection_length,
                    'intersection_percent': intersection_length / (span_data_list[i]['length'] +
                                                                   span_data_list[j]['length'] - intersection_length)
                })

    return intersections


if __name__ == '__main__':
    intersections = [
        {
            'start_index': 0,
            'length': 10
        },
        {
            'start_index': 5,
            'length': 10
        },
        {
            'start_index': 40,
            'length': 10
        },
    ]

    print(merge_intervals(intersections))