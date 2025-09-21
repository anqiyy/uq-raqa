'''
Script for generating hop combinations for MuSiQue dataset
'''
import json
import jsonlines
import numpy as np
from itertools import combinations

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

# change the paths accordingly
musique_path = "/afs/inf.ed.ac.uk/group/project/xwikis_msc/alex-yu/obqa/data/musique_data/musique_ans_v1.0_dev.jsonl"
annotated_path = "/afs/inf.ed.ac.uk/group/project/xwikis_msc/alex-yu/obqa/outputs/gemma-3-4b-it-musique-test-distil_t5_ctREAR3-score-qwen2-72b-it-awq-TIT.jsonl"

musique_data = load_file(musique_path)
annotated_musique = load_file(annotated_path)

musique_by_qid = {_d["id"]: _d for _d in musique_data}
mixed_data = []
for entry in annotated_musique:
    qid = entry["q_id"]
    # qid = entry["question_id"]

    musique_entry = musique_by_qid.get(qid)

    if not musique_entry:
        print(f"No match for q_id: {qid}. Skipping.")
        continue
    answerable = musique_entry["answerable"]
    if not answerable:
        continue
    # Extract hop passages from "question_decomposition"
    hops_idx = [hop["paragraph_support_idx"] for hop in musique_entry["question_decomposition"]]
    # print(hops_idx)
    hop_passages = [psg for psg in musique_entry["paragraphs"] if psg["idx"] in hops_idx] # or psg["is_supporting"]== True
    # print(hop_passages)

    # build combinations
    combined_passages = []
    for r in range(1, len(hop_passages) + 1):
        for combo in combinations(hop_passages, r):
            combined_title = " + ".join(p["title"] for p in combo)
            combined_text = "\n\n".join(p["paragraph_text"] for p in combo)
            combined_passages.append({
                "title": combined_title,
                "text": combined_text,
                "id": combined_title,  # or any unique identifier
                "score": "0.0",
                "hasanswer": True,
                'is_supporting': True
            })
    # print(len(hop_passages), len(combined_passages))
    # print(combined_passages)
    # print(entry["ctxs"])
    # break
    entry["ctxs"] = combined_passages + entry["ctxs"]
    mixed_data.append(entry)    
    

save_file_jsonl(mixed_data, "/afs/inf.ed.ac.uk/group/project/xwikis_msc/alex-yu/obqa/data/musique_data/musique-test-hop-combi.jsonl")