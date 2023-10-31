from wandb import Api
import json
import numpy as np
import scipy.stats as st


def main():
    api = Api()
    runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': args['tags']})
    runs = list(runs)

    data_ood = {}
    for run in runs:
        js_summary = run.summary._json_dict
        config = json.loads(run.json_config)

        domain = config['left_out_domain']['value'].split('/')[1][:-9]
        model = config['model']['value']
        seed = int(config['seed']['value'])
        if model != args['model']:
            continue
        if seed not in args['valid_seeds']:
            continue

        if domain not in data_ood.keys():
            data_ood[domain] = {
                'f1 block': [js_summary['ood_f1_block']],
                'f1 half': [js_summary['ood_f1_half_min']],
                'f1 one': [js_summary['ood_f1_one_min']],
                'accuracy block': [js_summary['ood_accuracy_blocks']],
                'accuracy half': [js_summary['ood_accuracy_half_min']],
                'accuracy one': [js_summary['ood_accuracy_one_min']],
            }
        else:
            data_ood[domain]['f1 block'].append(js_summary['ood_f1_block'])
            data_ood[domain]['f1 half'].append(js_summary['ood_f1_half_min'])
            data_ood[domain]['f1 one'].append(js_summary['ood_f1_one_min'])
            data_ood[domain]['accuracy block'].append(js_summary['ood_accuracy_blocks'])
            data_ood[domain]['accuracy half'].append(js_summary['ood_accuracy_half_min'])
            data_ood[domain]['accuracy one'].append(js_summary['ood_accuracy_one_min'])

    of = open('reklama_ood_results.csv', 'w+', encoding='utf-8')
    of.write('domain;')
    of.write(';'.join(list(data_ood['ctk'].keys())))
    of.write('\n')
    for domain, metrics in data_ood.items():
        of.write(f'{domain};')
        for metric, values in metrics.items():
            mean = np.mean(values)

            if len(values) > 1:
                conf95 = st.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=st.sem(values))
                of.write(f'{mean:.3f} ± {mean - conf95[0]:.3f};')
            else:
                of.write(f'{mean:.3f};')

        of.write('\n')

    of.close()


if __name__ == '__main__':
    args = {
        'tags': 'nofilter-v1.1',
        'model': 'Seznam/small-e-czech',
        # 'model': 'UWB-AIR/Czert-B-base-cased',
        'valid_seeds': [1, 2, 3, 4, 5]
    }

    main()
