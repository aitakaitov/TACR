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


def extend_k_tokens(token_classes):
    k = args['extend_k_tokens']
    in_span = False
    i = 0
    while i < len(token_classes):
        if token_classes[i] == 1 and not in_span:
            in_span = True
            go_back = False if i == 0 else True
            j = -1
            while go_back:
                if i + j < 0 or -j > k:
                    break
                token_classes[i + j] = 1
                j -= 1

        elif token_classes[i] == 1 and in_span:
            pass

        elif token_classes[i] == 0 and not in_span:
            pass

        elif in_span and token_classes[i] == 0:
            in_span = False
            go_forward = False if i == len(token_classes) - 1 else True
            j = 0
            while go_forward:
                if i + j > len(token_classes) - 1 or j == k:
                    i += j
                    break
                token_classes[i + j] = 1
                j += 1

        i += 1


def get_token_classes(sample, _type):
    span_classes = sample['classes_split']
    token_splits = sample['word_ids']
    token_classes = []
    for ts in token_splits:
        clss = 0
        if span_classes[ts][_type]:
            clss = 1
        token_classes.append(clss)

    if args['extend_k_tokens'] != 0:
        extend_k_tokens(token_classes)

    return token_classes


def fraction_hit_full_count(token_classes, attrs):
    new_attrs = [a for a in attrs]
    in_span = False
    span_length = 0
    hits = 0
    start = -1
    for i in range(len(token_classes)):
        if token_classes[i] == 1 and not in_span:
            in_span = True
            span_length += 1
            start = i
            if new_attrs[i] == 1:
                hits += 1
        elif token_classes[i] == 1 and in_span:
            span_length += 1
            if new_attrs[i] == 1:
                hits += 1
        elif token_classes[i] == 0 and not in_span:
            continue
        elif token_classes[i] == 0 and in_span:
            in_span = False

            if hits / span_length >= args['fraction_hit_full_count']:
                for j in range(start, start + span_length):
                    new_attrs[j] = 1

            start = -1
            span_length = 0
            hits = 0

    if in_span:
        if hits / span_length >= args['fraction_hit_full_count']:
            for j in range(start, start + span_length):
                new_attrs[j] = 1

    return new_attrs


def get_tp_fp_fn(sample, attrs, _type):
    tp = 0
    fn = 0
    fp = 0

    token_classes = get_token_classes(sample, _type)

    if args['fraction_hit_full_count']:
        attrs = fraction_hit_full_count(token_classes, attrs)

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

    pos_prec = []
    pos_rec = []
    neg_prec = []
    neg_rec = []
    for i, sample in enumerate(samples):
        if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'non_ads':
            prec, rec, f1 = evaluate_class_f1(sample, 0)
            if prec is not None:
                neg_prec.append(prec)
                neg_rec.append(rec)
            ap = avep(sample, 0)
            if ap is not None:
                neg_map += ap
                neg_map_count += 1
        if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'ads':
            prec, rec, f1 = evaluate_class_f1(sample, 1)
            if prec is not None:
                pos_prec.append(prec)
                pos_rec.append(rec)
            ap = avep(sample, 1)
            if ap is not None:
                pos_map += ap
                pos_map_count += 1

    print('k;method;posf1;posprec;posrec;posmap;negf1;negprec;negrec;negmap\n')
    if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'ads':
        prec = sum(pos_prec) / len(pos_prec)
        rec = sum(pos_rec) / len(pos_rec)
        f1 = 2 * prec * rec / (rec + prec)
        wandb.log({'positive_f1': f1, 'positive_precision': prec, 'positive_recall': rec, 'positive_map': pos_map / pos_map_count})
        print(f'Positive F1: {f1}')
        print(f'Positive Precision: {prec}')
        print(f'Positive Recall: {rec}')
        print(f'Positive MAP: {pos_map / pos_map_count}')

    if args['evaluate_class'] == 'both' or args['evaluate_class'] == 'non_ads':
        prec = sum(neg_prec) / len(neg_prec)
        rec = sum(neg_rec) / len(neg_rec)
        f1 = 2 * prec * rec / (rec + prec)
        wandb.log({'negative_f1': f1, 'negative_precision': prec, 'negative_recall': rec, 'negative_map': neg_map / neg_map_count})
        print(f'Negative F1: {f1}')
        print(f'Negative Precision: {prec}')
        print(f'Negative Recall: {rec}')
        print(f'Negative MAP: {neg_map / neg_map_count}')

    #print(f'{args["fraction_hit_full_count"]};method;{pos_f1 / f1_pos_count};{pos_prec / f1_pos_count};{pos_rec / f1_pos_count};{pos_map / pos_map_count};{neg_f1 / f1_neg_count};{neg_prec / f1_neg_count};{neg_rec / f1_neg_count};{neg_map / neg_map_count}')


def parse_list_to_ints(string):
    return [int(s) for s in string.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='sg25_keep_all_025.jsonl', type=str)
    parser.add_argument('--top_k_tokens', default=None, type=int)
    parser.add_argument('--top_p_tokens', default=5, type=int)
    parser.add_argument('--evaluate_class', default='both', type=str)
    parser.add_argument('--tags', default='', type=str)

    parser.add_argument('--extend_k_tokens', default=0, type=int)
    parser.add_argument('--fraction_hit_full_count', default=None, type=float)

    args = vars(parser.parse_args())

    split = args['file'].split('_')
    #wandb.init(config={**args, 'method': split[1], 'block_size': split[2], 'model': split[0]}, project='lrec-2024')
    wandb.init(config={**args, 'file': args['file']}, tags=None if args['tags'] is None else args['tags'].split(','), project='tacr-reklama')

    main()
