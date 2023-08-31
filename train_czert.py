import numpy as np
import torch
import transformers
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import wandb
import time

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from ood_verify import run_verify


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    results = {**accuracy_metric.compute(predictions=predictions, references=labels), **f1_metric.compute(predictions=predictions, references=labels)}
    wandb.log({"accuracy": results["accuracy"], "f1": results["f1"]})
    return results


def tokenize(examples):
    if 'intfloat' not in args['model']:
        return tokenizer(examples['text'], truncation=True, max_length=512)
    else:
        return tokenizer('query: ' + examples['text'], truncation=True, max_length=512)


def main():
    dataset = load_dataset('json', data_files=args['dataset_json_path'], split='train').map(tokenize)
    split_dataset = dataset.train_test_split(test_size=args['test_split_size'])
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
    train_dataset = train_dataset.shuffle(seed=42)

    config = transformers.AutoConfig.from_pretrained(args['model'])
    if args['dropout'] is not None:
        config.classifier_dropout = args['dropout']
        config.hidden_dropout_prob = args['dropout']
        config.attention_probs_dropout_prob = args['dropout']

    config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(args['model'], config=config)

    training_arguments = TrainingArguments(
        args['save_name'],
        evaluation_strategy='epoch',
        learning_rate=args['lr'],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=args['epochs'],
        weight_decay=1e-5,
        fp16=True,  # True,
        save_strategy='epoch',
        group_by_length=True
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8, max_length=512)

    trainer = Trainer(
        model,
        training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(args['save_name'])

    if ood_test:
        run_verify(tokenizer, model, args['ood_test_json_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, default=1, type=int)
    parser.add_argument('--model', required=True, default='UWB-AIR/Czert-B-base-cased', type=str)
    parser.add_argument('--lr', required=True, default=1e-5, type=float)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    parser.add_argument('--test_split_size', required=False, default=0.1, type=float)
    parser.add_argument('--dataset_json_path', required=True, default=None, type=str)
    parser.add_argument('--ood_test_json_path', required=False, default=None, type=str)
    parser.add_argument('--dropout', default=None, type=float, required=False)
    parser.add_argument('--save_name', required=True, type=str)

    args = vars(parser.parse_args())

    ood_test = args['ood_test_json_path'] is not None

    h = str(time.time_ns())
    wandb.init(project='tacr-reklama', entity='aitakaitov', tags=[h], config={
        'lr': args['lr'],
        'batch_size': args['batch_size'],
        'model': args['model'],
        'dataset': args['dataset_json_path'],
        'left_out_domain': args['dataset_json_path'][17:-9] if ood_test else None,
        'model_save': args['save_name']
    })

    tokenizer = AutoTokenizer.from_pretrained(args['model'], use_fast=False)
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric('f1')

    main()

