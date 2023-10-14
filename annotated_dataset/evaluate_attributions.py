import json
import argparse
import numpy as np
import wandb


def threshold_k(attributions):
    indexed_list = list(enumerate(attributions))
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [i for i, _ in indexed_list[:args['top_k_tokens']]]

    thresholded_attributions = []
    for i in range(len(attributions)):
        if i in top_k_indices:
            thresholded_attributions.append(1)
        else:
            thresholded_attributions.append(0)

    return thresholded_attributions


def threshold_p(attributions):
    percentile = np.percentile(attributions, 100 - args['top_p_tokens'])
    thresholded_attributions = []
    for attr in attributions:
        if attr < percentile:
            thresholded_attributions.append(0)
        else:
            thresholded_attributions.append(1)

    return thresholded_attributions


def get_token_classes(sample, _type):
    span_classes = sample['classes_split']
    token_splits = sample['word_ids']
    token_classes = []
    for ts in token_splits:
        clss = 0
        if span_classes[ts][_type]:
            clss = 1
        token_classes.append(clss)

    return token_classes


def get_tp_fp_fn(sample, attrs, _type):
    tp = 0
    fn = 0
    fp = 0

    token_classes = get_token_classes(sample, _type)
    control = 0
    for tc, attr in zip(token_classes, attrs):
        if attr == 1 and tc == 1:
            tp += 1
        elif attr == 0 and tc == 1:
            fn += 1
        elif attr == 1 and tc == 0:
            fp += 1

        if tc == 0:
            control += 1

    return tp, fp, fn, control


def token_f1(sample, attrs, _type):
    tp, fp, fn, control = get_tp_fp_fn(sample, attrs, _type)

    if control == len(attrs):
        return None, None, None

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_class_f1(sample, clss):
    if args['top_k_tokens']:
        attrs_thr = threshold_k(sample[f'{"pos" if clss == 1 else "neg"}_class_attrs'])
    else:
        attrs_thr = threshold_p(sample[f'{"pos" if clss == 1 else "neg"}_class_attrs'])

    precision, recall, f1 = token_f1(sample, attrs_thr, clss)

    return precision, recall, f1


def avep(sample, clss):
    indexed_attrs = list(enumerate(sample[f'{"pos" if clss == 1 else "neg"}_class_attrs']))
    indexed_attrs.sort(key=lambda x: x[1], reverse=True)
    # sorted_indices = [ia[0] for ia in indexed_attrs]
    token_classes = get_token_classes(sample, clss)

    if sum(token_classes) == 0:
        return None

    tp = 0
    fp = 0
    #fn = sum(token_classes)
    average_precision = 0
    for token_index in range(len(indexed_attrs)):
        if token_classes[indexed_attrs[token_index][0]] == 0:
            fp += 1
            continue
        else:
            tp += 1
            average_precision += tp / (tp + fp)

    return average_precision / sum(token_classes)


def main():
    with open(args['file'], 'r', encoding='utf-8') as f:
        samples = f.readlines()
    samples = [json.loads(s) for s in samples]

    pos_f1 = pos_prec = pos_rec = neg_f1 = neg_prec = neg_rec = f1_pos_count = f1_neg_count = 0
    pos_map = neg_map = pos_map_count = neg_map_count = 0

    for i, sample in enumerate(samples):
        if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'non_ads':
            prec, rec, f1 = evaluate_class_f1(sample, 0)
            if prec is not None:
                neg_prec += prec
                neg_rec += rec
                neg_f1 += f1
                f1_neg_count += 1
            ap = avep(sample, 0)
            if ap is not None:
                neg_map += ap
                neg_map_count += 1
        if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'ads':
            prec, rec, f1 = evaluate_class_f1(sample, 1)
            if prec is not None:
                pos_prec += prec
                pos_rec += rec
                pos_f1 += f1
                f1_pos_count += 1
            ap = avep(sample, 1)
            if ap is not None:
                pos_map += ap
                pos_map_count += 1

    if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'ads':
        wandb.log({'positive_f1': pos_f1 / f1_pos_count, 'positive_precision': pos_prec / f1_pos_count, 'positive_recall': pos_rec / f1_pos_count, 'positive_map': pos_map / pos_map_count})
        print(f'Positive F1: {pos_f1 / f1_pos_count}')
        print(f'Positive Precision: {pos_prec / f1_pos_count}')
        print(f'Positive Recall: {pos_rec / f1_pos_count}')
        print(f'Positive MAP: {pos_map / pos_map_count}')

    if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'non_ads':
        wandb.log({'negative_f1': neg_f1 / f1_neg_count, 'negative_precision': neg_prec / f1_neg_count, 'negative_recall': neg_rec / f1_neg_count, 'negative_map': neg_map / neg_map_count})
        print(f'Negative F1: {neg_f1 / f1_neg_count}')
        print(f'Negative Precision: {neg_prec / f1_neg_count}')
        print(f'Negative Recall: {neg_rec / f1_neg_count}')
        print(f'Negative MAP: {neg_map / neg_map_count}')


def parse_list_to_ints(string):
    return [int(s) for s in string.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='keep_all_optimal.jsonl', type=str)
    parser.add_argument('--top_k_tokens', default=None, type=int)
    parser.add_argument('--top_p_tokens', default=1, type=int)
    parser.add_argument('--evaluate_class', default='both', type=str)
    args = vars(parser.parse_args())

    split = args['file'].split('_')
    wandb.init(config={**args, 'method': split[0], 'block_size': split[1], 'model': split[2], 'input_file': split[3]}, project='lrec-2024')

    main()
