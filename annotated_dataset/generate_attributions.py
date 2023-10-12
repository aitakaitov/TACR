import json
import argparse

import torch.cuda
import transformers

import attribution_methods as _attrs
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = 'attributions_html'
SOURCE_DIR = 'attribution_articles'
BLOCK_SIZE = 128


def format_attrs(attrs):
    """
    Preprocesses the attributions shape and removes cls and sep tokens
    :param attrs:
    :param sentence:
    :return:
    """
    if len(attrs.shape) == 3:
        attrs = torch.mean(attrs, dim=2)

    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(attrs) - 1]  # leave out cls and sep


def embed_input_ids(input_ids):
    """
    Prepares input embeddings and attention mask
    :param sentence:
    :return:
    """
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, torch.squeeze(input_ids).to(device)), 0).requires_grad_(True).to(device)
    return input_embeds


def generate_attributions(input_embeds, att_mask):
    ig_attrs = _attrs.ig_attributions(input_embeds, att_mask, 1, 0 * input_embeds, model, logit_fn, 30)
    ig_attrs = torch.squeeze(ig_attrs)
    ig_attrs = ig_attrs.mean(dim=1)  # average over the embedding attributions
    ig_attrs = format_attrs(ig_attrs)

    return ig_attrs


def split_text(sample):
    text_split = []
    split_classes = []
    text = sample['text']
    points = []
    for span in sample['positive_spans']:
        points.append((span['start_index'], 1))
        points.append((span['start_index'] + span['length'], 1))

    for span in sample['negative_spans']:
        points.append((span['start_index'], 0))
        points.append((span['start_index'] + span['length'], 0))

    points.sort(key=lambda x: x[0])

    in_pos = False
    in_neg = False
    last_point = 0
    for i in range(len(points)):
        if i == len(points) - 1:
            text_split.append(text[last_point:points[i][0]])
            split_classes.append((in_neg, in_pos))

            if points[i][0] != len(text):
                text_split.append(text[points[i][0]:])
                split_classes.append((False, False))
        elif i == 0:
            text_split.append(text[last_point:points[i][0]])
            split_classes.append((in_neg, in_pos))
            if points[i][1] == 0:
                in_neg = True
            else:
                in_pos = True
            last_point = points[i][0]
        else:
            text_split.append(text[last_point:points[i][0]])
            split_classes.append((in_neg, in_pos))
            if points[i][1] == 0:
                in_neg = not in_neg
            else:
                in_pos = not in_pos
            last_point = points[i][0]

    empty_indices = []
    for i in range(len(text_split)):
        if text_split[i] == '':
            empty_indices.append(i)

    empty_indices.sort(reverse=True)
    for i in empty_indices:
        del split_classes[i]
        del text_split[i]

    return text_split, split_classes


def tokenize_text(text_split):
    return tokenizer(text_split, is_split_into_words=True, add_special_tokens=False)


def split_into_blocks(encoding):
    length = len(encoding.input_ids)
    block_count = int(length / BLOCK_SIZE)
    if length % BLOCK_SIZE != 0:
        block_count += 1

    blocks = []
    for i in range(block_count):
        if i == block_count - 1:
            input_ids = [cls_token_index]
            input_ids.extend(encoding.input_ids[i * BLOCK_SIZE:])
            input_ids.append(sep_token_index)
            blocks.append({
                'input_ids': torch.tensor([input_ids], dtype=torch.int),
                'attention_mask': torch.tensor([[1 for _ in range(len(input_ids))]], dtype=torch.int)
            })
        else:
            input_ids = [cls_token_index]
            input_ids.extend(encoding.input_ids[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE])
            input_ids.append(sep_token_index)
            blocks.append({
                'input_ids': torch.tensor([input_ids], dtype=torch.int),
                'attention_mask': torch.tensor([[1 for _ in range(len(input_ids))]], dtype=torch.int)
            })

    return blocks


def sample_attributions(encoding):
    blocks = split_into_blocks(encoding)

    positive_complete = []
    negative_complete = []
    for block in blocks:
        input_embeds = embed_input_ids(block['input_ids'])
        attention_mask = block['attention_mask'].to(device)

        if args['method'] == 'random':
            positive_attributions = _attrs.random_attributions(block['input_ids'])
            negative_attributions = _attrs.random_attributions(block['input_ids'])
        elif args['method'] == 'gradsxi':
            positive_attributions = _attrs.gradient_attributions(input_embeds, attention_mask, 1, model, logit_fn, True)
            negative_attributions = _attrs.gradient_attributions(input_embeds, attention_mask, 0, model, logit_fn, True)
        elif args['method'] == 'grads':
            positive_attributions = _attrs.gradient_attributions(input_embeds, attention_mask, 1, model, logit_fn, False)
            negative_attributions = _attrs.gradient_attributions(input_embeds, attention_mask, 0, model, logit_fn, False)
        elif 'sg' in args['method']:
            samples = int(args['method'][2:])
            positive_attributions = _attrs.sg_attributions(input_embeds, attention_mask, 1, model, logit_fn, samples, stdev_spread=0.15).to(device)
            positive_attributions *= input_embeds
            negative_attributions = _attrs.sg_attributions(input_embeds, attention_mask, 0, model, logit_fn, samples, stdev_spread=0.15).to(device)
            negative_attributions *= input_embeds
            positive_attributions = positive_attributions.to('cpu')
            negative_attributions = negative_attributions.to('cpu')
        elif 'ig' in args['method']:
            samples = int(args['method'][2:])
            negative_attributions = _attrs.ig_attributions(input_embeds, attention_mask, 0, 0 * input_embeds, model, logit_fn, steps=samples)
            positive_attributions = _attrs.ig_attributions(input_embeds, attention_mask, 1, 0 * input_embeds, model, logit_fn, steps=samples)

        positive_complete.extend(format_attrs(positive_attributions))
        negative_complete.extend(format_attrs(negative_attributions))

    return positive_complete, negative_complete


def run_eval(samples):
    tp = tn = fp = fn = 0
    for sample in samples:
        text = sample['text']
        encoded = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
        prediction = model(encoded.input_ids.to(device), encoded.attention_mask.to(device)).logits
        prediction = logit_fn(prediction)

        pred_class = int(torch.argmax(prediction))
        if pred_class == 1 and sample['label'] == 1:
            tp += 1
        elif pred_class == 1 and sample['label'] == 0:
            fp += 1
        elif pred_class == 0 and sample['label'] == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f'Classification F1: {2 * (precision * recall) / (precision + recall)}')


def main():
    with open(args['input_file'], 'r', encoding='utf-8') as f:
        samples = f.readlines()

    samples = [json.loads(s) for s in samples]
    # run_eval(samples)

    with open(args['output_file'], 'w+', encoding='utf-8') as f:
        for sample in tqdm(samples):
            # preprocessing to split the text according to spans
            text_split, split_classes = split_text(sample)
            encoding = tokenize_text(text_split)
            pos_attrs, neg_attrs = sample_attributions(encoding)

            f.write(json.dumps({
                'label': sample['label'],
                'text_split': text_split,
                'classes_split': split_classes,
                'word_ids': encoding.word_ids(),
                'pos_class_attrs': pos_attrs,
                'neg_class_attrs': neg_attrs
            }) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../sec-final-e2')
    parser.add_argument('--input_file', type=str, default='keep_all_pars_only.jsonl')#'05_mf0.6_ma2_poF_ms2_misl2_masl1500_s250_mI_scB.jsonl')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--method', type=str, default='random')
    parser.add_argument('--block_size', type=int, default=510)
    args = vars(parser.parse_args())

    print(args['method'])

    if args['output_file'] is None:
        args['output_file'] = f'{args["method"]}_bs{args["block_size"]}_{args["input_file"]}'

    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model']).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model'])

    embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data.to(device)
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id
    logit_fn = torch.nn.Softmax(dim=1)

    main()
    print()
    print()