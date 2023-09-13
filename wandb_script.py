from wandb import Api
import json
import numpy as np


def main():
    api = Api()
    runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': 'ood_testing'})
    runs = list(runs)

    data = {}

    for run in runs:
        js_summary = run.summary._json_dict
        domain = json.loads(run.json_config)['left_out_domain']['value'][14:]
        if domain not in data.keys():
            data[domain] = {'f1': [js_summary['ood_test_f1']], 'accuracy': [js_summary['ood_test_accuracy']]}
        else:
            data[domain]['f1'].append(js_summary['ood_test_f1'])
            data[domain]['accuracy'].append(js_summary['ood_test_accuracy'])

    of = open('wandb_stats.csv', 'w+', encoding='utf-8')
    of.write('domain;f1;accuracy\n')

    for domain, metrics in data.items():
        f1 = metrics['f1']
        accuracy = metrics['accuracy']

        f1_std = float(np.std(f1))
        accuracy_std = float(np.std(accuracy))
        f1_avg = sum(f1) / len(f1)
        accuracy_avg = sum(accuracy) / len(accuracy)

        of.write('{};{:.4f} ± {:.4f};{:.4f} ± {:.4f}\n'.format(domain, f1_avg, f1_std, accuracy_avg, accuracy_std))


        #of.write(f'{domain};{f1_avg} ± {f1_std};{accuracy_avg} ± {accuracy_std}\n')

    of.close()


if __name__ == '__main__':
    main()