'''
Script to generate hop combinations for hop passages for HotpotQA
'''
from random import randrange
from itertools import combinations
import json
import jsonlines
import numpy as np
import argparse


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_hotpotqa', type=str, required=True)    
    parser.add_argument('--path_to_rag', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    args = parser.parse_args()

    hotpotqa_data = load_file(args.path_to_hotpotqa) # hotpot/hotpot_train_v1.1.json
    rag_data = load_file(args.path_to_rag) # contriever-msmarco/hotpotqa-train.jsonl

    # --- Build qid index from hotpot ---
    hotpot_by_qid = {_d["_id"]: _d for _d in hotpotqa_data}

    # --- Replace passages ---
    mixed_data = []
    for rag_entry in rag_data:
        qid = rag_entry["q_id"]
        hotpot_entry = hotpot_by_qid.get(qid)

        if not hotpot_entry:
            print(f"No HotpotQA match for q_id: {qid}. Skipping.")
            continue
        
        # Extract candidate passages from HotpotQA's context
        converted_context = {title: " ".join(sentences) for title, sentences in hotpot_entry.get("context", [])}
        hop_passages = {title for title, _ in hotpot_entry.get("supporting_facts", [])}
        replace_count = len(hop_passages)

        hotpot_hop_passages = [{
            "title": title,
            "text": converted_context.get(title),
            "id": title,
            "score": "0.0",
            "hasanswer": False
        } for title in hop_passages]
        # print(distractor_passages)
        # build combinations
        combined_passages = []
        for r in range(1, len(hotpot_hop_passages) + 1):
            for combo in combinations(hotpot_hop_passages, r):
                combined_title = " + ".join(p["title"] for p in combo)
                combined_text = "\n\n".join(p["text"] for p in combo)
                combined_passages.append({
                    "title": combined_title,
                    "text": combined_text,
                    "id": combined_title,  # or any unique identifier
                    "score": "0.0",
                    "hasanswer": True
                })

        # Pick passages to replace in RAG
        rag_ctxs = rag_entry.get("ctxs", [])

        rag_entry["ctxs"] = combined_passages
        mixed_data.append(rag_entry)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in mixed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
