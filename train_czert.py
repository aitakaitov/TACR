import numpy as np
import torch
import tqdm
import transformers
from datasets import load_dataset, Dataset
from datasets import load_metric
from datasets import disable_caching
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
import wandb
import time

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
disable_caching()

from ood_verify import run_verify


def compute_metrics(p):
    predictions, labels = p

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=1)
    results = {**accuracy_metric.compute(predictions=predictions, references=labels), **f1_metric.compute(predictions=predictions, references=labels)}
    wandb.log({"accuracy": results["accuracy"], "f1": results["f1"]})
    return results


def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512 if not args['whole_document'] else None)


def split_into_blocks(encoding, cls_token_index, sep_token_index, block_size):
    length = len(encoding.input_ids)
    block_count = int(length / block_size)
    if length % block_size != 0:
        block_count += 1

    blocks = []
    for i in range(block_count):
        if i == block_count - 1:
            input_ids = [cls_token_index]
            input_ids.extend(encoding.input_ids[i * block_size:])
            input_ids.append(sep_token_index)
            blocks.append({
                'input_ids': input_ids,
                'attention_mask': [1 for _ in range(len(input_ids))]
            })
        else:
            input_ids = [cls_token_index]
            input_ids.extend(encoding.input_ids[i * block_size: (i + 1) * block_size])
            input_ids.append(sep_token_index)
            blocks.append({
                'input_ids': input_ids,
                'attention_mask': [1 for _ in range(len(input_ids))]
            })

    return blocks


def prepare_dataset_whole_docs(dataset):
    cls_token_index = tokenizer.cls_token_id
    sep_token_index = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 1

    new_dataset = []
    i = 0
    for sample in tqdm.tqdm(dataset):
        encoding = tokenizer(sample['text'], add_special_tokens=False)
        blocks = split_into_blocks(encoding, cls_token_index, sep_token_index, 510)
        if len(blocks) == 0:
            continue
        for block in blocks:
            new_dataset.append({**block, 'label': sample['label'], 'document_id': i})
        i += 1

    return Dataset.from_list(new_dataset)


def main():
    config = transformers.AutoConfig.from_pretrained(args['model'])
    if args['dropout'] is not None:
         config.classifier_dropout = args['dropout']
         config.hidden_dropout_prob = args['dropout']
         config.attention_probs_dropout_prob = args['dropout']

    config.num_labels = 2
    config.output_attentions = False
    config.output_hidden_states = False
    config.pooler_output = False

    model = AutoModelForSequenceClassification.from_pretrained(args['model'], config=config).to('cuda')

    if args['dataset_json_path']:
        if not args['whole_document']:
            dataset = load_dataset('json', data_files=args['dataset_json_path'], split='train').map(tokenize)
            split_dataset = dataset.train_test_split(test_size=args['test_split_size'], seed=42)
            train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
            train_dataset = train_dataset.shuffle(seed=42)
        else:
            dataset = load_dataset('json', data_files=args['dataset_json_path'], split='train')
            dataset = prepare_dataset_whole_docs(dataset)
            split_dataset = dataset.train_test_split(test_size=args['test_split_size'], seed=42)
            train_dataset, test_dataset = split_dataset['train'], split_dataset['test']
            train_dataset = train_dataset.shuffle(seed=42)

        training_arguments = TrainingArguments(
            args['save_name'],
            evaluation_strategy='epoch',
            do_eval=True,
            learning_rate=args['lr'],
            per_device_train_batch_size=args['batch_size'],
            per_device_eval_batch_size=args['batch_size'],
            num_train_epochs=args['epochs'],
            weight_decay=1e-5,
            fp16=True,  # True,
            save_strategy='epoch',
            group_by_length=True,
            eval_accumulation_steps=128 if 'barticzech' in args['model'] else None,
            gradient_accumulation_steps=args['gradient_acc_steps'],
            seed=args['seed'],
        )

        data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8, max_length=512)

        if args['lr_scheduler'] == 'linear_decay':
            trainer = Trainer(
                model,
                training_arguments,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
        else:
            total_steps = len(train_dataset) * args['epochs'] / (args['gradient_acc_steps'] * args['batch_size'])
            warmup_steps = int(args['warmup_steps'] * total_steps)

            if args['lr_scheduler'] == 'warmup_decay':
                optimizer = torch.optim.AdamW(params=model.parameters(), lr=args['lr'], weight_decay=1e-5)
                scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                         num_training_steps=total_steps)

            trainer = Trainer(
                model,
                training_arguments,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                optimizers=(optimizer, scheduler)
            )

        trainer.train()
        trainer.save_model(args['save_name'])

    if ood_test:
        test_dataset = load_dataset('json', data_files=args['ood_test_json_path'], split='train').map(tokenize)
        test_dataset = prepare_dataset_whole_docs(test_dataset)
        run_verify(model, test_dataset)


def parse_bool(s):
    return s.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', required=False, default=1, type=int)
    parser.add_argument('--model', required=True, default='UWB-AIR/Czert-B-base-cased', type=str)
    parser.add_argument('--lr', required=True, default=1e-5, type=float)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    parser.add_argument('--test_split_size', required=False, default=0.15, type=float)
    parser.add_argument('--dataset_json_path', required=False, default=None, type=str)
    parser.add_argument('--ood_test_json_path', required=False, default=None, type=str)
    parser.add_argument('--dropout', default=None, type=float, required=False)
    parser.add_argument('--save_name', required=True, type=str)
    parser.add_argument('--tags', required=False, default=None, type=str)
    parser.add_argument('--domain', required=False, default=None, type=str)
    parser.add_argument('--seed', default=-1, required=False, type=int)
    parser.add_argument('--whole_document', default=True, type=parse_bool)
    parser.add_argument('--gradient_acc_steps', default=1, type=int)
    parser.add_argument('--lr_scheduler', default='linear_decay', type=str)
    parser.add_argument('--warmup_steps', default=0.1, type=float)

    args = vars(parser.parse_args())

    if args['seed'] == -1:
        torch.seed()
    else:
        torch.manual_seed(args['seed'])

    ood_test = args['ood_test_json_path'] is not None

    domain_dict = {}
    if args['domain'] is not None:
        domain_dict = {'domain': args['domain']}

    seed_dict = {}
    if args['dataset_json_path'] is None:
        seed_dict['seed'] = args['seed']

    wandb.init(project='tacr-reklama', entity='aitakaitov', tags=None if args['tags'] is None else args['tags'].split(','), config={
        'lr': args['lr'],
        'batch_size': args['batch_size'],
        'model': args['model'],
        'dataset': args['dataset_json_path'],
        'left_out_domain': args['dataset_json_path'] if ood_test else None,
        'model_save': args['save_name'],
        **args,
        **domain_dict,
        **seed_dict
    })

    tokenizer = AutoTokenizer.from_pretrained(args['model'], use_fast=True)
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric('f1')

    main()

