import jsonlines
import json
import numpy as np
from glob import glob
from sklearn import metrics

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_file_json(data, fp):
    with open(fp, 'w') as f:
        json.dump(data, f)        

def auroc(y_true, y_score, verbose=False):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr[1:] - fpr[1:])
    optimal_threshold = thresholds[1:][optimal_idx]
    if verbose:
        print('Optimal threshold', optimal_threshold)
    del thresholds
    if verbose:
        return metrics.auc(fpr, tpr), optimal_threshold
    else:
        return metrics.auc(fpr, tpr)


def sigmoid(z):
    return 1/(1 + np.exp(-z))


predFiles = glob('individual_eval/*.jsonl')
models = ['gemma-2-9b-it', 'gemma-2-27b-it', 'llama-3.1-8b-instruct', 'mistral-7B-instruct-v03']
datasets = ['nq', 'tqa', 'webq', 'squad', 'popqa3k', 'refunq']

selected_400_idxs = {}
for d in datasets:
    f = open(f'individual_eval/{d}_test_eval_idxs.txt')
    ids = [l.strip().split('at id=`')[1].split('`.')[0] for l in f.readlines()]
    selected_400_idxs[d] = ids


results = {'gemma-2-9b-it': {},
           'gemma-2-27b-it': {},
           'llama-3.1-8b-instruct': {},
           'mistral-7B-instruct-v03': {}}
for mdFile in predFiles:
    print(mdFile)
    md = load_file(mdFile)
    for d in datasets:
        if f'-{d}-' in mdFile: 
            dataset = d
            break
    for m in models:
        if f'{m}-' in mdFile: 
            model = m
            break        
    print(dataset, len(selected_400_idxs[dataset]), selected_400_idxs[dataset][:5])
    md = [item for item in md if str(item['q_id']) in selected_400_idxs[dataset]]
    #mdPreds = [(1- sigmoid(c['acc_LM-nli_pred'])) for item in md for c in item['ctxs'][:5]]
    mdPreds = [c['ppl'] for item in md for c in item['ctxs'][:5]]
    mdAccuracies = [1- int(c['acc_LM']) for item in md for c in item['ctxs'][:5]]
    metric_i, opt_th = auroc(mdAccuracies, mdPreds, verbose=True)
    results[model][dataset] = f"{round(metric_i, 2)}"
    print(f"{round(metric_i, 2)}")


for m in results.keys():
    tab_line = m 
    avg = 0
    for d in datasets:
        tab_line += ' & ' + results[m][d]
        avg += float(results[m][d])
    tab_line += ' & ' + f"{round(avg / len(datasets), 2)}" 
    print(tab_line)
