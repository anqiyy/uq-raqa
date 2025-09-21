import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
import jsonlines
import numpy as np
import argparse
import os

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

def get_hop_analysis_hotpotqa(data):
    '''
    For HotPotQA
    '''
    # Categorize each question
    category_counts = Counter()
    category_index = {'Single Passage':[], 'Only Multi-Passage':[], 'Unanswerable':[]}
    for entry in data:
        single_correct = False
        multi_correct = False

        for ctx in entry['ctxs']:
            acc = ctx.get('acc_LM', 0)
            is_combo = '+' in ctx['id'] # For concatenated passages 

            if acc:
                if is_combo:
                    multi_correct = True
                else:
                    single_correct = True

        if single_correct:
            category_counts['Single Passage'] += 1
            category_index['Single Passage'].append(entry['q_id'])
        elif multi_correct:
            category_counts['Only Multi-Passage'] += 1
            category_index['Only Multi-Passage'].append(entry['q_id'])
        else:
            category_counts['Unanswerable'] += 1
            category_index['Unanswerable'].append(entry['q_id'])

    # Bar chart
    labels = list(category_counts.keys())
    values = [category_counts[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title('Minimum Required Passages per Question')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    return category_counts, category_index

def get_hop_analysis_musique(data):
    """
    For MuSiQue
    """
    category_counts = Counter()
    category_index = {'Less than n hop Passages':[], 'Only Multi-Passage':[], 'Unanswerable':[]}
    for entry in data:
        single_correct = False
        multi_correct = False
        hops = entry["q_id"][0] # number of "+" we want to find in a title
        # print(entry["q_id"], hops)
        # break
        for ctx in entry['ctxs']:
            acc = ctx.get('acc_LM', 0)
            # print(ctx["id"])
            is_combo = False
            if ctx["id"].count("+") == int(hops)-1:
                is_combo = True # Find the ctx where text has all hops present
            # is_combo = '+' in ctx['id'] # For concatenated passages 
            # print("acc: ", acc, "is_combo: ", is_combo)
            if acc == 1:
                if is_combo:
                    multi_correct = True
                else:
                    single_correct = True
        if single_correct:
                # print(entry["q_id"], multi_correct)
                category_counts['Less than n hop Passages'] += 1
                category_index['Less than n hop Passages'].append(entry['q_id'])
        elif multi_correct:
            # print(entry["q_id"])
            category_counts['Only Multi-Passage'] += 1
            category_index['Only Multi-Passage'].append(entry['q_id'])
        else:
            category_counts['Unanswerable'] += 1
            category_index['Unanswerable'].append(entry['q_id'])
        
    # Bar chart
    labels = list(category_counts.keys())
    values = [category_counts[k] for k in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title('Minimum Required Passages per Question')
    plt.ylabel('Number of Questions')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()
    return category_counts, category_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    print(os.getcwd())


    
    data = load_file(args.input_file)
    # if input file is hotpotQA
    hp_dev_category_counts, hp_dev_category_index = get_hop_analysis_hotpotqa(data)
    # if input file is MuSiQue
    mq_dev_category_counts, mq_dev_category_index = get_hop_analysis_musique(data)



if __name__ == "__main__":
    main()
