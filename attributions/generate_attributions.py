import json
import os
import argparse
import numpy as np

import torch.cuda
import transformers

import attribution_methods_custom as _attrs


device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = 'attributions_html'
SOURCE_DIR = 'attribution_articles'


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
    #ig_attrs *= input_embeds
    ig_attrs = torch.squeeze(ig_attrs)
    ig_attrs = ig_attrs.mean(dim=1)  # average over the embedding attributions
    ig_attrs = format_attrs(ig_attrs)

    return ig_attrs #{
        #'Integrated Gradients, n=20, baseline=zero': ig_attrs
    #}


def get_percentile(attrs):
    return (np.percentile([abs(a) for a in attrs], args['upper_percentile']),
            np.percentile([abs(a) for a in attrs], args['lower_percentile']))


def preprocess_attrs(attrs, tokens):
    attrs = token_to_word_attributions(attrs, tokens)
    attrs = color_spans(attrs, tokens)
    return attrs


def color_spans(attrs, tokens):
    RANGE = 1
    for i in range(len(attrs) - (RANGE + 1)):
        if attrs[i] != 0 and attrs[i + RANGE + 1] != 0:
            for j in range(1, RANGE + 1):
                if attrs[i + j] == 0:
                    attrs[i + j] = attrs[i + j - 1]

    return attrs


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


def export_article_attributions(input_ids, article, attributions):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    max_allowed_attr, min_allowed_attr = get_percentile(attributions)
    for i in range(len(attributions)):
        if abs(attributions[i]) > max_allowed_attr:
            attributions[i] = 0.0

    bound = max(abs(min(attributions)), abs(max(attributions)))

    for i in range(len(attributions)):
        if attributions[i] < min_allowed_attr:
            attributions[i] = 0
        attributions[i] /= bound

    attributions = preprocess_attrs(attributions, tokens)

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

    html = ''
    for token, attr in zip(tokens, attributions):
        if abs(attr) == bound:
            attr = 0
        r, g, b = get_color(attr)
        if '##' in token:
            token = token[2:]
            html += f"<span style='color:rgb({r},{g},{b})'>{token}</span>"
        elif token in [',', '.', ':', '?', '!', '-']:
            html += f"<span style='color:rgb({r},{g},{b})'>{token}</span>"
        else:
            html += f" <span style='color:rgb({r},{g},{b})'>{token}</span>"

    html = "<html><body style='font-size: 20px;font-family: Arial, Helvetica, sans-serif;'>" + html + '</body></html>'
    return html


def process_article(article, domain, filename):
    input_ids, input_embeds, att_mask = embed_input_ids(article['data'])
    attributions = generate_attributions(input_embeds, att_mask)
    html = export_article_attributions(torch.squeeze(input_ids).tolist()[1:-1], article, attributions)

    with open(f'{OUTPUT_DIR}/{domain}/{filename}.html', 'w+', encoding='utf-8') as f:
        f.write(html)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    domains = os.listdir(SOURCE_DIR)

    for domain in domains:
        print(domain)
        os.makedirs(f'{OUTPUT_DIR}/{domain}', exist_ok=True)
        articles = os.listdir(f'{SOURCE_DIR}/{domain}')

        for article in articles:
            with open(f'{SOURCE_DIR}/{domain}/{article}', 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
            process_article(data, domain, article)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--upper_percentile', type=int, default=95)
    parser.add_argument('--lower_percentile', type=int, default=60)
    args = vars(parser.parse_args())

    model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model']).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args['model'])

    OUTPUT_DIR += f'_up{args["upper_percentile"]}_lp{args["lower_percentile"]}'

    # TODO check model type
    embeddings = model.electra.base_model.embeddings.word_embeddings.weight.data.to(device)
    logit_fn = torch.nn.Softmax(dim=1)

    main()