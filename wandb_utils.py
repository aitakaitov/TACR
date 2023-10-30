import wandb
import json
import numpy as np
import scipy.stats as st


top_p = 20
#models = ['czert-final-wholedoc-0', 'czert-final-wholedoc-1', 'czert-final-wholedoc-2', 'czert-final-wholedoc-3', 'czert-final-wholedoc-4']

#datasets = {'075-3-3_union.jsonl': {}, '075-3-3_intersection.jsonl': {}, '075-3-3_keep_all.jsonl': {}, '06-2-2_union.jsonl': {}, '06-2-2_intersection.jsonl': {}}

#datasets = {'066-2-2_keep_all.jsonl': {}}

datasets = {
    '066-2-2-union': {},
    '075-3-3-union': {}
}

for k, d in datasets.items():
    d['ig25'] = {}
    d['ig50'] = {}
    d['sg25'] = {}
    d['sg50'] = {}
    d['random'] = {}
    for m in d.keys():
        d[m]['pos_map'] = []
        d[m]['pos_f1'] = []
        d[m]['pos_prec'] = []
        d[m]['pos_rec'] = []


api = wandb.Api(api_key='bb99cb2b7077af36587e0415e9ea6b87a1f0013b')
runs = api.runs('aitakaitov/tacr-reklama', filters={'tags': 'nofilter-v1-attrs'})
for run in runs:
    a = 0
    config = json.loads(run.json_config)
    summary = run.summary._json_dict

    if config['top_p_tokens']['value'] != top_p:
        continue

    model = config['file']['value'].split('_')[2].split('/')[1]
    if 'sec' in model:
        continue
    #if 'tags' not in config.keys():
    #    continue
    #if 'tags' in config.keys():
    #    continue

    #if '_keep_all.jsonl' not in config['file']['value'] or '_bs' in config['file']['value']:
    #    continue
    #if config['tags']['value'] not in datasets.keys():
    #    continue

    #dataset = config['tags']['value']
    #dataset = '066-2-2_keep_all_pars_only.jsonl'
    dataset = '066-2-2-union' if '06-2-2' in config['file']['value'] else '075-3-3-union'
    #method = config['method']['value']
    method = config['file']['value'].split('_')[3]

    if '100' in method or '75' in method:
        continue

    pos_map = summary['positive_map']
    pos_f1 = summary['positive_f1']
    pos_prec = summary['positive_precision']
    pos_rec = summary['positive_recall']

    datasets[dataset][method]['pos_map'].append(pos_map)
    datasets[dataset][method]['pos_f1'].append(pos_f1)
    datasets[dataset][method]['pos_prec'].append(pos_prec)
    datasets[dataset][method]['pos_rec'].append(pos_rec)

for dataset, methods in datasets.items():
    with open(f'result.csv', 'w+', encoding='utf-8') as f:
        f.write('method;positive map;positive f1;positive prec;positive rec\n')
        for method, data in methods.items():
            f.write(f'{method}')
            for metric, values in data.items():
                mean = np.mean(values)
                conf95 = st.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=st.sem(values))
                f.write(f';{mean:.3f} ± {mean - conf95[0]:.3f}')
            f.write('\n')
