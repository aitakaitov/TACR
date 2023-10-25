import copy
import json
import os
import argparse
import random
import time

import bs4

random.seed(42)


def load_csv(path):
    res = {}
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            domain, count = line.split(';')
            res[domain] = int(count)
    
    return res


def write_to_json(obj, path):
    with open(path, 'w+') as f:
        f.write(json.dumps(obj))


def load_from_json(path):
    with open(path, 'r') as f:
        obj = json.loads(f.read())
    
    return obj


def load_counts():
    return load_csv('art_counts.csv'), load_csv('ad_counts.csv')


def filter_domains(art_counts, ad_counts):
    print('Filtering domains')
    art_domains = set([d for d, c in art_counts.items()])
    ad_domains = set([d for d, c in ad_counts.items()])

    valid_domains = list(art_domains.intersection(ad_domains))
    print(f'Valid domains: {valid_domains} ({len(valid_domains)})')

    if args['leave_out_domain'] is not None and args['leave_out_domain'] in valid_domains:
        valid_domains.remove(args['leave_out_domain'])

    for domain in valid_domains:
        if domain in args['invalid_domains']:
            valid_domains.remove(domain)

    art_counts_new = {d: art_counts[d] for d in valid_domains}
    ad_counts_new = {d: ad_counts[d] for d in valid_domains}

    return art_counts_new, ad_counts_new


def subsample_ads(ad_counts):
    print('Subsampling ads')
    ad_counts['ctk'] = 3000
    ad_counts['drbna'] = 3000
    return ad_counts


def match_arts_to_ads(art_counts, ad_counts):
    print('Reducing number of articles')
    for domain, art_count in art_counts.items():
        ad_count = ad_counts[domain]
        expected_arts = args['arts_per_ad'] * ad_count
        art_counts[domain] = min(art_count, expected_arts)
    
    print(f'New article counts: {art_counts}')
    print(f'Total articles: {sum([c for d, c in art_counts.items()])}')
    print(f'Total ads: {sum([c for d, c in ad_counts.items()])}')
    return art_counts


def select_files(tld, counts):
    print(f'Selecting from \'{tld}\'')
    selected_files = {}
    for domain, count in counts.items():
        print(f'Processing \'{domain}\'')
        dir_path = os.path.join(tld, domain, args['source_dir_type'])
        files = os.listdir(dir_path)
        files = [str(os.path.join(tld, domain, args['source_dir_type'], file)) for file in files]
        random.shuffle(files)
        selected_files[domain] = files[:count]

    return selected_files


def trim_text_start_length(text):
    MIN_TOKENS = args['trim_length']
    tag_texts = text.split('\n')
    start = -1
    for i, tag in enumerate(tag_texts):
        if len(tag.split()) > MIN_TOKENS:
            start = i
            break

    new_text = ''
    for i in range(start, len(tag_texts)):
        new_text += tag_texts[i] + '\n'

    return new_text


def trim_text(text):
    if args['trim_text'] is None:
        return text
    elif args['trim_text'] == 'start_length':
        return trim_text_start_length(text)
    else:
        print(f'{args["trim_text"]} not valid')
        exit(-1)


def get_text_from_html(html):
    soup = bs4.BeautifulSoup(html)
    return soup.get_text()


def write_dataset(art_files, ad_files, dataset_name):
    with open(os.path.join(args["folder"], f'{dataset_name}.json'), 'w+') as f:
        for domain in art_files.keys():
            for file in art_files[domain]:
                data = load_from_json(file)

                if 'html' in args['source_dir_type']:
                    data['data'] = get_text_from_html(data['data'])

                data['data'] = trim_text(data['data'])
                f.write(json.dumps(
                    {
                        'text': data['data'],
                        'label': 0,
                        'file': file
                    }
                ) + '\n')

        for domain in ad_files.keys():
            for file in ad_files[domain]:
                data = load_from_json(file)

                if 'html' in args['source_dir_type']:
                    data['data'] = get_text_from_html(data['data'])

                data['data'] = trim_text(data['data'])
                f.write(json.dumps(
                    {
                        'text': data['data'],
                        'label': 1,
                        'file': file
                    }
                ) + '\n')


def create_dataset(art_counts, ad_counts, dataset_name, subsample_ads_: bool):
    print(f'Creating dataset {dataset_name}')
    ad_counts_local = copy.copy(ad_counts)

    if subsample_ads_:
        ad_counts_local = subsample_ads(ad_counts_local)

    art_counts_local = copy.copy(art_counts)
    art_counts_local = match_arts_to_ads(art_counts_local, ad_counts_local)

    art_files = select_files('art_pages', art_counts_local)
    ad_files = select_files('ad_pages', ad_counts_local)

    print(f'--- TOTAL ARTICLES: {sum([len(v) for k, v in art_files.items()])}')
    print(f'--- TOTAL ADS: {sum([len(v) for k, v in ad_files.items()])}')

    write_to_json({'art_files': art_files, 'ad_files': ad_files}, os.path.join(args['folder'], f'{dataset_name}_files.json'))

    write_dataset(art_files, ad_files, dataset_name)


def main():
    dataset_name = 'ads_dataset'
    suffix = '' if args['leave_out_domain'] is None else f'_{args["leave_out_domain"]}-out'

    art_counts_original, ad_counts_original = load_counts()
    art_counts, ad_counts = filter_domains(art_counts_original, ad_counts_original)

    create_dataset(art_counts, ad_counts, f'{dataset_name}_full{suffix}', subsample_ads_=False)

    if args['leave_out_domain'] is None:
        create_dataset(art_counts, ad_counts, f'{dataset_name}_subsampled{suffix}', subsample_ads_=True)
    else:
        art_counts = {args['leave_out_domain']: art_counts_original[args['leave_out_domain']]}
        ad_counts = {args['leave_out_domain']: ad_counts_original[args['leave_out_domain']]}
        create_dataset(art_counts, ad_counts, f'{dataset_name}_full_{args["leave_out_domain"]}-only', subsample_ads_=False)


def filter_domains_complete(art_counts, ad_counts):
    print('Filtering domains')
    art_domains = set([d for d, c in art_counts.items()])
    ad_domains = set([d for d, c in ad_counts.items()])

    valid_domains = list(art_domains.intersection(ad_domains))
    print(f'Domains pre-filtering: {valid_domains}')

    for domain in valid_domains:
        if domain in args['invalid_domains']:
            valid_domains.remove(domain)

    print(f'Valid domains: {valid_domains} ({len(valid_domains)})')
    art_counts_new = {d: art_counts[d] for d in valid_domains}
    ad_counts_new = {d: ad_counts[d] for d in valid_domains}

    return art_counts_new, ad_counts_new


def remove_a_domain_temp(art_files, ad_files, domain):
    art_files_new = copy.deepcopy(art_files)
    ad_files_new = copy.deepcopy(ad_files)

    art_files_domain = {domain: copy.deepcopy(art_files_new[domain])}
    ad_files_domain = {domain: copy.deepcopy(ad_files_new[domain])}

    art_files_new.pop(domain)
    ad_files_new.pop(domain)

    return art_files_new, ad_files_new, art_files_domain, ad_files_domain


def get_domain_files(domain):
    art_dir_path = os.path.join('art_pages', domain, args['source_dir_type'])
    ad_dir_path = os.path.join('ad_pages', domain, args['source_dir_type'])

    art_files = os.listdir(art_dir_path)
    ad_files = os.listdir(ad_dir_path)

    expected_art_count = len(ad_files) * args['arts_per_ad']
    actual_art_count = min(expected_art_count, len(art_files))

    random.shuffle(art_files)
    art_files = art_files[:actual_art_count]

    art_files = [str(os.path.join('art_pages', domain, args['source_dir_type'], file)) for file in art_files]
    ad_files = [str(os.path.join('ad_pages', domain, args['source_dir_type'], file)) for file in ad_files]

    return {domain: art_files}, {domain: ad_files}


def complete_ood():
    art_counts_original, ad_counts_original = load_counts()
    art_counts_original, ad_counts_original = filter_domains_complete(art_counts_original, ad_counts_original)
    art_counts_matched = match_arts_to_ads(art_counts_original, ad_counts_original)

    art_files = select_files('art_pages', art_counts_matched)
    ad_files = select_files('ad_pages', ad_counts_original)

    # dump the files as a training set
    write_to_json({'art_files': art_files, 'ad_files': ad_files}, os.path.join(args['folder'], f'training_full_files.json'))
    write_dataset(art_files, ad_files, f'training_full')

    # generate an 'out' file for each domain by cutting from the training pool
    for domain in ad_files.keys():
        print('processing cut domain ' + domain)
        art_files_cut, ad_files_cut, art_files_domain, ad_files_domain = remove_a_domain_temp(art_files, ad_files, domain)
        print('writing -out')
        write_to_json({'art_files': art_files_cut, 'ad_files': ad_files_cut},
                      os.path.join(args['folder'], f'{domain}-out_files.json'))
        write_dataset(art_files_cut, ad_files_cut, f'{domain}-out')

        print('writing -only')
        write_to_json({'art_files': art_files_domain, 'ad_files': ad_files_domain},
                      os.path.join(args['folder'], f'{domain}-only_files.json'))
        write_dataset(art_files_domain, ad_files_domain, f'{domain}-only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arts_per_ad', default=2, type=int)
    parser.add_argument('--leave_out_domain', default=None)
    parser.add_argument('--folder', required=True, default='', type=str)
    parser.add_argument('--trim_text', required=False, default=None)
    parser.add_argument('--trim_length', required=False, default=0, type=int)
    parser.add_argument('--invalid_domains', required=False, default='expres,forbes', type=str)
    parser.add_argument('--random_seed', required=False, default=False, type=bool)
    parser.add_argument('--source_dir_type', required=True, type=str)
    args = vars(parser.parse_args())

    if args['random_seed']:
        random.seed(time.process_time_ns())

    args['invalid_domains'] = args['invalid_domains'].split(',')
    print(f'Invalid domains: {args["invalid_domains"]}')

    os.makedirs(args['folder'], exist_ok=True)

    if args['leave_out_domain'] != 'all':
        main()
    else:
        complete_ood()
