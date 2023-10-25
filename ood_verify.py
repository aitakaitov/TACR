import json
import random

import wandb

random.seed(42)

import transformers
import torch
import argparse

from transformers import BertTokenizer
from datasets import load_metric
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ads(dataset_file=None):
    ads = []
    domains = []
    labels = []
    with open(args['dataset_file'] if dataset_file is None else dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if args['max_samples'] is not None:
            if len(lines) > args['max_samples']:
                random.shuffle(lines)
                lines = lines[:args['max_samples']]

        for line in lines:
            sample = json.loads(line)
            ads.append(sample)
            domain = sample['file'].split('/')[1]
            domains.append(domain)
            labels.append(sample['label'])

    return ads, domains, labels


def get_predictions(model, tokenizer: BertTokenizer, ad_samples, max_position_embeddings):
    predictions = []
    for sample in tqdm(ad_samples):
        encoded = tokenizer(sample['text'], return_tensors='pt', max_length=max_position_embeddings, truncation=True)
        logits = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device)).logits.to('cpu')
        predictions.append(int(torch.argmax(logits, dim=1)))

    return predictions


def print_results(predictions, labels, domains):
    f1 = load_metric('f1')
    acc = load_metric('accuracy')

    wandb.log({
         'ood_test_f1': f1.compute(predictions=predictions, references=labels)['f1'],
         'ood_test_accuracy': acc.compute(predictions=predictions, references=labels)['accuracy']
    })


def get_document_predictions_one_min(doc_block_preds):
    doc_predictions = [0 for _ in range(len(doc_block_preds.keys()))]
    for doc_index, preds in doc_block_preds.items():
        doc_predictions[doc_index] = 1 if sum(preds) > 0 else 0

    return doc_predictions


def get_document_predictions_half_min(doc_block_preds):
    doc_predictions = [0 for _ in range(len(doc_block_preds.keys()))]
    for doc_index, preds in doc_block_preds.items():
        doc_predictions[doc_index] = 1 if sum(preds) > len(preds) / 2.0 else 0

    return doc_predictions


def run_verify(model, test_dataset):
    block_predictions = []
    block_labels = []
    for sample in tqdm(test_dataset):
        logits = model(input_ids=sample['input_ids'].to(device), attention_mask=sample['attention_mask'].to(device)).logits.to('cpu')
        block_predictions.append(int(torch.argmax(logits, dim=1)))
        block_labels.append(sample['label'])

    document_block_predictions = {}
    document_labels = {}
    for prediction, sample in zip(block_predictions, test_dataset):
        if sample['document_id'] not in document_block_predictions.keys():
            document_block_predictions[sample['document_id']] = [prediction]
            document_labels[sample['document_id']] = sample['label']
        else:
            document_block_predictions[sample['document_id']].append(prediction)

    labels = [0 for _ in range(len(document_labels.keys()))]
    for doc_index, label in document_labels.items():
        labels[doc_index] = label

    doc_preds_one_min = get_document_predictions_one_min(document_block_predictions)
    doc_preds_half_min = get_document_predictions_half_min(document_block_predictions)

    f1 = load_metric('f1')
    acc = load_metric('accuracy')

    wandb.log({
        'ood_f1_block': f1.compute(predictions=block_predictions, references=block_labels)['f1'],
        'ood_f1_one_min': f1.compute(predictions=doc_preds_one_min, references=labels)['f1'],
        'ood_f1_half_min': f1.compute(predictions=doc_preds_half_min, references=labels)['f1'],
        'ood_accuracy_blocks': f1.compute(predictions=block_predictions, references=block_labels)['accuracy'],
        'ood_accuracy_one_min': acc.compute(predictions=doc_preds_one_min, references=labels)['accuracy'],
        'ood_accuracy_half_min': acc.compute(predictions=doc_preds_half_min, references=labels)['accuracy'],
    })


def main():
    ad_samples, domains, labels = load_ads()
    predictions = get_predictions(model, tokenizer, ad_samples, model.config.max_position_embeddings)
    print_results(predictions, labels, domains)


def check_bool(inp):
    return inp.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str, default=None, required=False)
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--max_samples', default=None, type=int, required=False)
    args = vars(parser.parse_args())

    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_save'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_save']).to(device)
    config = transformers.AutoConfig.from_pretrained(args['model_save'])

    wandb.init(config={**vars(config), 'max_samples': args['max_samples'], 'chatgpt': args['use_chatgpt'], 'dataset': args['dataset_file']}, project='tacr-reklama', tags=['ood_500'])

    main()
