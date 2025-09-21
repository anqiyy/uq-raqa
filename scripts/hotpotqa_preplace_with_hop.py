import json
import jsonlines
import argparse
from random import randrange, sample

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
        json.dump(data, f, indent=2)    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Mix RAG data with HotpotQA hop passages.')
    parser.add_argument('--file_input', type=str, required=True, help='Input RAG JSONL file path')
    parser.add_argument('--file_output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--replace_mode', type=str, choices=['top', 'first_and_random'], default='first_and_random',
                      help='Passage replacement mode: "top" or "first_and_random"')
    parser.add_argument('--hotpotqa_file', type=str, required=True, help='HotpotQA hotpotqa file path')
    parser.add_argument('--replace_count', type=int, default=2, help='Number of passages to replace')
    
    args = parser.parse_args()

    # Load RAG file
    rag_data = load_file(args.file_input)

    # Load hotpotqa file
    hotpotqa_data = load_file(args.hotpotqa_file)

    # Build qid index from hotpotqa
    hotpotqa_by_qid = {_d["_id"]: _d for _d in hotpotqa_data}

    # Process data
    mixed_data = []
    for rag_entry in rag_data:
        qid = rag_entry["q_id"]
        hotpotqa_entry = hotpotqa_by_qid.get(qid)

        if not hotpotqa_entry:
            print(f"No hotpotqa match for q_id: {qid}. Skipping.")
            continue
        
        # Extract candidate passages from hotpotqa's context
        converted_context = {title: " ".join(sentences) for title, sentences in hotpotqa_entry.get("context", [])}
        hop_passages = {title for title, _ in hotpotqa_entry.get("supporting_facts", [])}

        hotpotqa_passages = [{
            "title": title,
            "text": converted_context.get(title),
            "id": title,
            "score": "0.0",
            "hasanswer": True
        } for title in hop_passages]

        if len(hotpotqa_passages) < args.replace_count:
            print(f"Not enough hotpotqa passages for q_id: {qid}")
            continue

        # Pick passages to replace in RAG
        rag_ctxs = rag_entry.get("ctxs", [])
        if len(rag_ctxs) < args.replace_count:
            print(f"Not enough original ctxs for q_id: {qid}")
            continue
        
        if args.replace_mode == "top":
            replace_indices = list(range(args.replace_count))
        elif args.replace_mode == "first_and_random":
            n_to_replace = len(hotpotqa_passages)
            replace_indices = [0]
            replace_indices += sample([1, 2, 3, 4], n_to_replace - 1)
            print(replace_indices, n_to_replace)

        # Replace passages
        # new_ctxs = rag_ctxs[:5]
        for i, d_passage in zip(replace_indices, hotpotqa_passages):
            rag_ctxs[i] = d_passage
            # print(rag_ctxs[i])
            # break

        rag_entry["ctxs"] = rag_ctxs
        print(len(rag_entry["ctxs"]))
        # print(rag_entry["ctxs"])
        mixed_data.append(rag_entry)
        # break

    # Save output
    with open(args.file_output, 'w', encoding='utf-8') as f:
        for entry in mixed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()