import json
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
        for line in f.readlines():
            sample = json.loads(line)
            ads.append(sample)
            domain = sample['file'].split('/')[1]
            domains.append(domain)
            labels.append(sample['label'])

    return ads, domains, labels


def get_predictions(model, tokenizer: BertTokenizer, ad_samples):
    predictions = []
    for sample in tqdm(ad_samples):
        encoded = tokenizer(sample['text'], return_tensors='pt', max_length=512, truncation=True)
        logits = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device)).logits.to('cpu')
        predictions.append(int(torch.argmax(logits, dim=1)))

    return predictions


def print_results(predictions, labels, domains):
    f1 = load_metric('f1')
    acc = load_metric('accuracy')

    results = {**f1.compute(predictions=predictions, references=labels),
               **acc.compute(predictions=predictions, references=labels)}

    print(results)


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_save'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_save']).to(device)

    ad_samples, domains, labels = load_ads()
    predictions = get_predictions(model, tokenizer, ad_samples)
    print_results(predictions, labels, domains)


def run_verify(tokenizer, model, dataset_file):
    import wandb

    ad_samples, domains, labels = load_ads(dataset_file)
    predictions = get_predictions(model, tokenizer, ad_samples)

    f1 = load_metric('f1')
    acc = load_metric('accuracy')

    wandb.log({
        'ood_test_f1': f1.compute(predictions=predictions, references=labels)['f1'],
        'ood_test_accuracy': acc.compute(predictions=predictions, references=labels)['accuracy']
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str, required=True)
    parser.add_argument('--dataset_file', type=str, required=True)
    args = vars(parser.parse_args())

    main()