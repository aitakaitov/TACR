import re

from annotated_dataset.similarity_utils import character_count_similarity_index


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

    return {
        'start_index': index,
        'length': span_length
    } if not report_multiples else {
        'start_index': index,
        'length': span_length,
        'count_simple': count_simple,
        'count_sim': count_sim
    }


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
