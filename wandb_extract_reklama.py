from wandb import Api
import json
import numpy as np


def main():
    api = Api()
    runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': 'nofilter-v1'})
    runs = list(runs)

    data = {}

    for run in runs:
        js_summary = run.summary._json_dict
        domain = json.loads(run.json_config)['left_out_domain']['value'][20:-9]
        model = json.loads(run.json_config)['_name_or_path']['value']
        if model == "Seznam/small-e-czech":
            continue

        if domain not in data.keys():
            data[domain] = {
                'f1 block': [js_summary['ood_f1_block']],
                'f1 half': [js_summary['ood_f1_half_min']],
                'f1 one': [js_summary['ood_f1_one_min']],
                'accuracy block': [js_summary['ood_accuracy_blocks']],
                'accuracy half': [js_summary['ood_accuracy_half_min']],
                'accuracy one': [js_summary['ood_accuracy_one_min']],
            }
        else:
            data[domain]['f1 block'].append(js_summary['ood_f1_block'])
            data[domain]['f1 half'].append(js_summary['ood_f1_half_min'])
            data[domain]['f1 one'].append(js_summary['ood_f1_one_min'])
            data[domain]['accuracy block'].append(js_summary['ood_accuracy_blocks'])
            data[domain]['accuracy half'].append(js_summary['ood_accuracy_half_min'])
            data[domain]['accuracy one'].append(js_summary['ood_accuracy_one_min'])

    of = open('reklama_stats.csv', 'w+', encoding='utf-8')
    of.write('domain;')
    of.write(';'.join(list(data['ctk'].keys())))
    of.write('\n')
    for domain, metrics in data.items():
        of.write(f'{domain};')
        for metric, values in metrics.items():
            avg = sum(values) / len(values)
            std = float(np.std(values))
            #of.write('{:.3f} ± {:.3f};'.format(avg, std))
            of.write(f'{avg:.3f};')

        of.write('\n')

    of.close()


if __name__ == '__main__':
    main()