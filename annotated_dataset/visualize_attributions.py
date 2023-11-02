import json
import os
import argparse
import numpy as np

import torch.cuda
import transformers


device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = 'attributions_html'


def get_percentile(attrs):
    return (np.percentile([abs(a) for a in attrs], args['upper_percentile']),
            np.percentile([abs(a) for a in attrs], args['lower_percentile']))


# def preprocess_attrs(attrs, tokens):
#     attrs = token_to_word_attributions(attrs, tokens)
#     attrs = color_spans(attrs, tokens)
#     return attrs
#
#
# def color_spans(attrs, tokens):
#     RANGE = 1
#     for i in range(len(attrs) - (RANGE + 1)):
#         if attrs[i] != 0 and attrs[i + RANGE + 1] != 0:
#             for j in range(1, RANGE + 1):
#                 if attrs[i + j] == 0:
#                     attrs[i + j] = attrs[i + j - 1]
#
#     return attrs


def token_to_word_attributions(attrs, tokens):
    _max = -1
    count = 0
    for i in range(1, len(attrs)):
        if tokens[i][:2] == '##':
            if count == 0:
                count = 2
                _max = attrs[i - 1]
                if attrs[i] > _max:
                    _max = attrs[i]
            else:
                count += 1
                if attrs[i] > _max:
                    _max = attrs[i]
        else:
            if count != 0:
                for j in range(1, count + 1):
                    attrs[i - j] = _max
                _max = -1
                count = 0
            else:
                continue

    return attrs


def preprocess(attributions):
    max_allowed_attr, min_allowed_attr = get_percentile(attributions)
    for i in range(len(attributions)):
        if abs(attributions[i]) > max_allowed_attr:
            attributions[i] = 0.0

    bound = max(abs(min(attributions)), abs(max(attributions)))

    for i in range(len(attributions)):
        if attributions[i] < min_allowed_attr:
            attributions[i] = 0
        attributions[i] /= bound

    return bound


def get_color(a):
    if a > 0:
        r = int(128 * a) + 127
        g = 128 - int(64 * a)
        b = 128 - int(64 * a)
    else:
        r = 128
        b = 128
        g = 128

    return r, g, b


def get_body(sample):
    text_tokenized = []
    for span in sample['text_split']:
        text_tokenized.append(tokenizer.tokenize(span))
    tokens = []
    [tokens.extend(t) for t in text_tokenized]

    bound = preprocess(sample['pos_class_attrs'])

    html = '<b>Atribuce:</b><br />'
    colored = []
    for token, attr in zip(tokens, sample['pos_class_attrs']):
        if abs(attr) == bound:
            attr = 0
        if attr != 0:
            attr = 1
        r, g, b = get_color(attr)
        if '##' in token:
            token = token[2:]
            colored.append(f"<span style='color:rgb({r},{g},{b})'>{token}</span>")
        elif token in [',', '.', ':', '?', '!', '-']:
            colored.append(f"<span style='color:rgb({r},{g},{b})'>{token}</span>")
        else:
            colored.append(f" <span style='color:rgb({r},{g},{b})'>{token}</span>")

    # generate text for annotations
    span_start = 0
    for span, classes in zip(text_tokenized, sample['classes_split']):
        span_tokens = len(span)
        if classes[1]:
            colored = colored[:span_start] + [f"<span style='background-color:rgb({128},{255},{128})'>"] + colored[span_start:span_start+span_tokens] + ['</span>'] + colored[span_start+span_tokens:]
            span_start += span_tokens + 2
        else:
            span_start += span_tokens

    html += ''.join(colored)

    return html


def get_header(sample, i):
    return (f'<a href="{i+1}.html">dalsi</a>'
            f'<br />'
            f'<a href="{i-1}.html">predchozi</a>'
            f'<br />'
            f'<span>label: <b>{"pozitivni" if sample["label"] == 1 else "negativni"}</b></span>'
            f'<br />')


def visualize(sample, i):
    html = "<html><body style='font-size: 20px;font-family: Arial, Helvetica, sans-serif;'>"
    html += get_header(sample, i)
    html += get_body(sample)

    html += '</body></html>'
    return html


def main():
    with open(args['input_file'], 'r', encoding='utf-8') as f:
        samples = f.readlines()
    for i in range(len(samples)):
        samples[i] = json.loads(samples[i])

    os.mkdir(OUTPUT_DIR)
    for i, sample in enumerate(samples):
        html = visualize(sample, i)
        with open(f'{OUTPUT_DIR}/{i}.html', 'w+', encoding='utf-8') as f:
            f.write(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Seznam/small-e-czech')
    parser.add_argument('--input_file', type=str, default='v1_sg75.jsonl')
    parser.add_argument('--upper_percentile', type=int, default=95)
    parser.add_argument('--lower_percentile', type=int, default=60)
    args = vars(parser.parse_args())

    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model'])

    # TODO check model type
    # embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data.to(device)
    # logit_fn = torch.nn.Softmax(dim=1)

    main()
