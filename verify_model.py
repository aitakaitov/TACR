import json
import transformers
import torch
import argparse

from transformers import BertTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ads():
    ads = []
    domains = []
    labels = []
    with open('annotated_dataset/dataset_annotated.jsonl', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sample = json.loads(line)
            ads.append(sample)
            file = sample['file']
            domains.append(file)
            label = sample['label']
            labels.append(label)

    return ads, domains, labels


def get_predictions(model, tokenizer: BertTokenizer, ad_samples):
    predictions = []
    for sample in ad_samples:
        encoded = tokenizer(sample['text'], return_tensors='pt', max_length=512, truncation=True)
        logits = model(input_ids=encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device)).logits.to('cpu')
        predictions.append(int(torch.argmax(logits, dim=1)))

    return predictions


def print_results(predictions, files, labels):
    with open('output.csv', 'w+', encoding='utf-8') as f:
        f.write('prediction;label;file\n')
        for p, l, fi in zip(predictions, labels, files):
            f.write(f'{p};{l};{fi}\n')


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model_save'])
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_save']).to(device)

    ad_samples, files, labels = load_ads()
    predictions = get_predictions(model, tokenizer, ad_samples)
    print_results(predictions, files, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save', type=str, required=True)
    args = vars(parser.parse_args())

    main()