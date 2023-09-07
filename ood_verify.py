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

    results = {**f1.compute(predictions=predictions, references=labels),
               **acc.compute(predictions=predictions, references=labels)}

    print(results)


def run_verify(tokenizer, model, dataset_file):
    ad_samples, domains, labels = load_ads(dataset_file)
    predictions = get_predictions(model, tokenizer, ad_samples, model.config.max_position_embeddings)

    f1 = load_metric('f1')
    acc = load_metric('accuracy')

    wandb.log({
        'ood_test_f1': f1.compute(predictions=predictions, references=labels)['f1'],
        'ood_test_accuracy': acc.compute(predictions=predictions, references=labels)['accuracy']
    })


def main():
    ad_samples, domains, labels = load_ads()
    predictions = get_predictions(model, tokenizer, ad_samples, model.config.max_position_embeddings)
    print_results(predictions, labels, domains)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str, required=True)
    parser.add_argument('--dataset_file', type=str, required=True)
    parser.add_argument('--max_samples', default=None, type=int, required=False)
    args = vars(parser.parse_args())

    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_save'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_save']).to(device)
    config = transformers.AutoConfig.from_pretrained(args['model_save'])

    wandb.init(config={**vars(config), 'max_samples': args['max_samples']})

    main()
