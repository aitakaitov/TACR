import json
import os
import argparse
import numpy as np

import torch.cuda
import transformers

import attribution_methods as _attrs


device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = 'attributions_html'
SOURCE_DIR = 'attribution_articles'
BLOCK_SIZE = 510

def format_attrs(attrs):
    """
    Preprocesses the attributions shape and removes cls and sep tokens
    :param attrs:
    :param sentence:
    :return:
    """
    if len(attrs.shape) == 2 and attrs.shape[0] == 1:
        attrs = torch.squeeze(attrs)

    attrs_list = attrs.tolist()
    return attrs_list[1:len(attrs) - 1]  # leave out cls and sep


def embed_input_ids(text):
    """
    Prepares input embeddings and attention mask
    :param sentence:
    :return:
    """
    encoded = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
    attention_mask = encoded.data['attention_mask'].to(device)
    input_embeds = torch.unsqueeze(torch.index_select(embeddings, 0, torch.squeeze(encoded.data['input_ids']).to(device)), 0).requires_grad_(True).to(device)

    return encoded.data['input_ids'], input_embeds, attention_mask


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
                'input_ids': input_ids,
                'attention_mask': [1 for _ in range(len(input_ids))]
            })
        else:
            input_ids = [cls_token_index]
            input_ids.extend(encoding.input_ids[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE])
            input_ids.append(sep_token_index)
            blocks.append({
                'input_ids': input_ids,
                'attention_mask': [1 for _ in range(len(input_ids))]
            })

    return blocks


def sample_attributions(encoding):
    blocks = split_into_blocks(encoding)


def main():
    with open(args['input_file'], 'r', encoding='utf-8') as f:
        samples = f.readlines()

    for sample in samples:
        # preprocessing to split the text according to spans
        sample = json.loads(sample)
        text_split, split_classes = split_text(sample)
        encoding = tokenize_text(text_split)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../sec-final-e2')
    parser.add_argument('--input_file', type=str, default='test_dataset.jsonl')
    parser.add_argument('--method', type=str, default='ig')
    args = vars(parser.parse_args())

    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model']).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model'])

    embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data.to(device)
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id
    logit_fn = torch.nn.Softmax(dim=1)

    main()