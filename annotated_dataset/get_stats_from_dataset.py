import json


def stats(fi):
    with open(fi, 'r', encoding='utf-8') as f:
        samples = f.readlines()

    samples = [json.loads(s) for s in samples]

    ad_docs = 0
    total_docs = 0
    doc_lengths = []
    ad_spans = 0
    art_spans = 0
    positive_span_lengths = []
    negative_span_lengths = []
    for sample in samples:
        if sample['label'] == 1:
            ad_docs += 1
        total_docs += 1

        doc_lengths.append(len(sample['text'].split()))
        ad_spans += len(sample['positive_spans'])
        art_spans += len(sample['negative_spans'])
        for span in sample['positive_spans']:
            text = sample['text'][span['start_index']:span['start_index'] + span['length']]
            positive_span_lengths.append(len(text.split()))

        for span in sample['negative_spans']:
            text = sample['text'][span['start_index']:span['start_index'] + span['length']]
            negative_span_lengths.append(len(text.split()))

    print(fi)
    print(f'Total docs: {total_docs}')
    print(f'Ad docs: {ad_docs} ({ad_docs / total_docs})')
    print(f'Avg length: {sum(doc_lengths) / len(doc_lengths)}')
    print(f'Total spans: {ad_spans + art_spans}')
    print(f'Ad spans: {ad_spans} ({ad_spans / (ad_spans + art_spans)})')
    print(f'Art spans: {art_spans} ({art_spans / (ad_spans + art_spans)})')
    print(f'Avg span length: {(sum(positive_span_lengths) + sum(negative_span_lengths)) / (len(positive_span_lengths) + len(negative_span_lengths))}')
    print(f'Avg ad span length: {sum(positive_span_lengths) / len(positive_span_lengths)}')
    print(f'Avg art span length: {sum(negative_span_lengths) / len(negative_span_lengths)}')
    print()
    print()


for file in ['unfiltered_union.jsonl', 'unfiltered_keep_all.jsonl', 'unfiltered_intersection.jsonl', '06-2-2_keep_all.jsonl', '06_2_2_union.jsonl', '06_2_2_intersection.jsonl', '075-3-3_keep_all.jsonl', '075-3-3_union.jsonl', '075-3-3_intersection.jsonl']:
    stats(file)
