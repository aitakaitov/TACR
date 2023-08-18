import json
import transformers
import torch
import argparse

from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ads():
    ads = []
    domains = []
    with open('ads_dataset_full.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sample = json.loads(line)
            if sample['label'] == 1:
                ads.append(sample)
                domain = sample['file'].split('/')[1]
                domains.append(domain)

    return ads, domains


def get_predictions(model, tokenizer: BertTokenizer, ad_samples):
    predictions = []
    for sample in ad_samples:
        encoded = tokenizer(sample['text'], return_tensors='pt', max_length=512, truncation=True)
        logits = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device)).logits.to('cpu')
        predictions.append(int(torch.argmax(logits, dim=1)))

    return predictions


def print_results(predictions, domains):
    domain_info = {d: [0, 0] for d in list(set(domains))}

    for pred, domain in zip(predictions, domains):
        domain_info[domain][pred] += 1

    for domain, info in domain_info.items():
        print(f'{domain}:\t {sum(info)} predictions, {info[1]} correct, {info[0]} incorrect')


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_save'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_save']).to(device)

    ad_samples, domains = load_ads()
    predictions = get_predictions(model, tokenizer, ad_samples)
    print_results(predictions, domains)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str, required=True)
    args = vars(parser.parse_args())

    main()