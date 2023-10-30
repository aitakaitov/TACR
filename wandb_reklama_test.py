from wandb import Api
import json
import numpy as np
import scipy.stats as st


def main():
    api = Api()
    runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': args['tags']})
    runs = list(runs)

    data_test = {
        '2-66': {
            'f1 block': [],
            'f1 half': [],
            'f1 one': [],
            'accuracy block': [],
            'accuracy half': [],
            'accuracy one': [],
        },
        '3-75': {
            'f1 block': [],
            'f1 half': [],
            'f1 one': [],
            'accuracy block': [],
            'accuracy half': [],
            'accuracy one': [],
        }
    }
    for run in runs:
        js_summary = run.summary._json_dict
        config = json.loads(run.json_config)

        tags = run.tags
        if '3-75' not in tags and '2-66' not in tags:
            continue

        dataset = '3-75' if '3-75' in tags else '2-66'

        model = config['model']['value']
        if args['model'] not in model:
            continue
        seed = int(config['seed']['value'])
        if seed not in args['valid_seeds']:
            continue

        data_test[dataset]['f1 block'].append(js_summary['ood_f1_block'])
        data_test[dataset]['f1 half'].append(js_summary['ood_f1_half_min'])
        data_test[dataset]['f1 one'].append(js_summary['ood_f1_one_min'])
        data_test[dataset]['accuracy block'].append(js_summary['ood_accuracy_blocks'])
        data_test[dataset]['accuracy half'].append(js_summary['ood_accuracy_half_min'])
        data_test[dataset]['accuracy one'].append(js_summary['ood_accuracy_one_min'])

    for dataset, data in data_test.items():
        of = open(f'reklama_test_results_{dataset}.csv', 'w+', encoding='utf-8')
        of.write(';'.join(list(data.keys())))
        of.write('\n')
        for metric, values in data.items():
            mean = np.mean(values)
            conf95 = st.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=st.sem(values))
            of.write(f'{mean:.3f} ± {mean - conf95[0]:.3f};')
        of.write('\n')

    of.close()


if __name__ == '__main__':
    args = {
        'tags': 'nofilter-v1-final',
        'model': 'sec',
        # 'model': 'UWB-AIR/Czert-B-base-cased',
        'valid_seeds': [1, 2, 3, 4, 5]
    }

    main()
