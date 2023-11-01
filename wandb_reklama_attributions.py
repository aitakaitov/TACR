import wandb
import numpy as np
import scipy.stats as st


def main():
    dataset_results = {}

    api = wandb.Api()
    runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': args['tags']})
    for run in runs:
        # config = json.loads(run.json_config)
        summary = run.summary._json_dict

        if run.config['top_p_tokens'] != args['top_p']:
            continue

        if args['model'] not in run.config['file']:
            continue

        dataset = run.config['dataset']
        if dataset not in dataset_results.keys():
            dataset_results[dataset] = {}

        method = run.config['method']
        if method not in dataset_results[dataset].keys():
            dataset_results[dataset][method] = {
                'pos_map': [],
                'pos_f1': [],
                'pos_prec': [],
                'pos_rec': [],
            }

        # if '100' in method or '75' in method:
        #    continue

        pos_map = summary['positive_map']
        pos_f1 = summary['positive_f1']
        pos_prec = summary['positive_precision']
        pos_rec = summary['positive_recall']

        dataset_results[dataset][method]['pos_map'].append(pos_map)
        dataset_results[dataset][method]['pos_f1'].append(pos_f1)
        dataset_results[dataset][method]['pos_prec'].append(pos_prec)
        dataset_results[dataset][method]['pos_rec'].append(pos_rec)

    for dataset, methods in dataset_results.items():
        with open(f'attr_results_{dataset}.csv', 'w+', encoding='utf-8') as f:
            f.write('method;positive map;positive f1;positive prec;positive rec\n')
            for method, data in methods.items():
                f.write(f'{method}')
                for metric, values in data.items():
                    mean = np.mean(values)
                    if len(values) > 1:
                        conf95 = st.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=st.sem(values))
                        f.write(f';{mean:.3f} ± {mean - conf95[0]:.3f}')
                    else:
                        f.write(f';{mean:.3f}')
                f.write('\n')


if __name__ == '__main__':
    args = {
        'top_p': 20,
        'tags': 'nofilter-v1.1-attrs',
        'model': 'sec'
    }
    main()