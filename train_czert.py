import numpy as np
import torch
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import wandb
import time


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    results = accuracy_metric.compute(predictions=predictions, references=labels) | f1_metric.compute(predictions=predictions, references=labels)
    wandb.log({"accuracy": results["accuracy"], "f1": results["f1"]})
    return results


def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)


def main():

    dataset = load_dataset('json', data_files=args['dataset_json_path'], split='train[:200]')\
         .map(tokenize)
    dataset = dataset.shuffle(seed=42)

    split_dataset = dataset.train_test_split(test_size=args['test_split_size'], shuffle=True, seed=42)
    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    model = AutoModelForSequenceClassification.from_pretrained(args['model'], num_labels=2)

    training_arguments = TrainingArguments(
        f'{args["model"]}_{args["lr"]}',
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

    trainer.save_model(h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, default=1, type=int)
    parser.add_argument('--model', required=True, default='UWB-AIR/Czert-B-base-cased', type=str)
    parser.add_argument('--lr', required=True, default=1e-5, type=float)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    parser.add_argument('--test_split_size', required=False, default=0.1, type=float)
    parser.add_argument('--dataset_json_path', required=True, type=str)

    args = vars(parser.parse_args())

    h = str(time.time_ns())
    wandb.init(project='tacr-reklama', entity='aitakaitov', tags=[h], config={
        'lr': args['lr'],
        'batch_size': args['batch_size'],
        'model': args['model'],
        'dataset': args['dataset_json_path']
    })

    tokenizer = AutoTokenizer.from_pretrained(args['model'])
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric('f1')

    main()
