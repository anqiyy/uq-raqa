from random import randrange, sample
import json
import jsonlines
import numpy as np


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


# Load RAG file
rag_path = "/afs/inf.ed.ac.uk/group/project/xwikis_msc/alex-yu/obqa/outputs/gemma-3-4b-it-musique-dev-distil_t5_ctREAR3-score-qwen2-72b-it-awq-TIT.jsonl"
rag_data = load_file(rag_path)

# Load hotpotqa file
path = "/afs/inf.ed.ac.uk/group/project/xwikis_msc/alex-yu/obqa/data/musique_data/musique_ans_v1.0_dev.jsonl"
musique_data = load_file(path)

# Build qid index 
mus_by_qid = {_d["id"]: _d for _d in musique_data}

# Process data
mixed_data = []
for rag_entry in rag_data:
    qid = rag_entry["q_id"]
    musique_entry = mus_by_qid.get(qid)

    if not musique_entry:
        print(f"No hotpotqa match for q_id: {qid}. Skipping.")
        continue
    
    # Extract hop passages from "question_decomposition"
    hops_idx = [hop["paragraph_support_idx"] for hop in musique_entry["question_decomposition"]]
    # print(hops_idx)
    hop_passages = [psg for psg in musique_entry["paragraphs"] if psg["idx"] in hops_idx] # or psg["is_supporting"]== True
    # print(hop_passages)

    replace_mode = "first_and_random"
    replace_count = 0
    # Pick passages to replace in RAG
    rag_ctxs = rag_entry.get("ctxs", [])
    
    if replace_mode == "top":
        replace_indices = list(range(replace_count))
    elif replace_mode == "first_and_random":
        n_to_replace = len(hop_passages)
        replace_indices = [0]
        replace_indices += sample([1, 2, 3, 4], n_to_replace - 1)
        print(replace_indices, n_to_replace)

    # Replace passages
    # new_ctxs = rag_ctxs[:5]
    for i, d_passage in zip(replace_indices, hop_passages):
        rag_ctxs[i] = d_passage
        # print(rag_ctxs[i])
        # break

    rag_entry["ctxs"] = rag_ctxs
    print(len(rag_entry["ctxs"]))
    # print(rag_entry["ctxs"])
    mixed_data.append(rag_entry)