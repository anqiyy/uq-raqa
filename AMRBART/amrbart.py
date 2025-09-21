import sys
sys.path.append("fine-tune")
import torch
import nltk
import json
import jsonlines
import argparse
import penman
from itertools import islice
from transformers import BartForConditionalGeneration
from model_interface.tokenization_bart import AMRBartTokenizer

from collections import defaultdict
from tqdm import tqdm

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        return [obj for obj in jsonl_f]

def load_file(input_fp):
    if input_fp.endswith(".json"):
        return json.load(open(input_fp))
    else:
        return load_jsonlines(input_fp)

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def preprocess_batch(texts, tokenizer, device, max_src_len=512, unified_input=True):
    amr_prefix = [
        tokenizer.bos_token_id,
        tokenizer.mask_token_id,
        tokenizer.eos_token_id,
        tokenizer.amr_bos_token_id
    ]
    amr_suffix = [tokenizer.amr_eos_token_id]

    encoded = []
    for text in texts:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_src_len - 3 if unified_input else max_src_len
            )
        if unified_input:
            input_ids = input_ids[: max_src_len - len(amr_prefix) - 1]
            input_ids = amr_prefix + input_ids + amr_suffix
        encoded.append(input_ids)

    return tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt").to(device)

def generate_amr_batch(batch_inputs, model, tokenizer, decode_amr=True):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_inputs["input_ids"],
            max_length=256,
            num_beams=2,  # use num_beams=1 for max speed
            decoder_start_token_id=tokenizer.amr_bos_token_id,
            eos_token_id=tokenizer.amr_eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded_strings = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []

    if decode_amr:
        for output_ids in outputs:
            ids = output_ids.tolist()
            ids[0] = tokenizer.bos_token_id
            ids = [
                tokenizer.eos_token_id if tid == tokenizer.amr_eos_token_id else tid
                for tid in ids if tid != tokenizer.pad_token_id
            ]
            graph, _, _ = tokenizer.decode_amr(ids)
            penman_str = penman.encode(graph)
            results.append(penman_str)
    else:
        results = decoded_strings

    return results

def main():
    parser = argparse.ArgumentParser(description='Running AMRBART for AMR Parsing (Batch Mode)')
    parser.add_argument('--file_input', type=str, required=True)
    parser.add_argument('--file_output', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--no_examples', type=int, required=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fields', nargs='+', required=True)  # e.g., ctxs"

    args = parser.parse_args()

    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = AMRBartTokenizer.from_pretrained(args.model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    rag_data = load_file(args.file_input)
    print(f"Loaded {len(rag_data)} examples")

    batch_size = args.batch_size
    device = model.device
    
    for field in args.fields:
        print(f"Processing field: {field}")
        all_texts = []
        meta = []

        for d_idx, d in enumerate(rag_data):
            for ctx_idx, ctx in enumerate(d[field]): # ctxs
                if isinstance(ctx, dict) and "text" in ctx:
                    all_texts.append(ctx["text"])
                    meta.append((d_idx, field, ctx_idx))


        #print(all_texts)
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_meta = meta[i:i + batch_size]

            inputs = preprocess_batch(batch_texts, tokenizer, device)
            graphs = generate_amr_batch(inputs, model, tokenizer, decode_amr=True)

            for (d_idx, field, ctx_idx), amr_graph in zip(batch_meta, graphs):
                if isinstance(ctx_idx, int):  # case for list of dicts like 'ctxs'
                    rag_data[d_idx][field][ctx_idx]["sssp_output"] = amr_graph
                elif ctx_idx is None:  # single field
                    rag_data[d_idx][field + "_amr"] = amr_graph
                else:  # nested dict
                    rag_data[d_idx].setdefault(field + "_amr", {})[ctx_idx + "_amr"] = amr_graph

    # Step 3: Now write final output
    with open(args.file_output, 'w', encoding='utf-8') as f_out:
        for d in rag_data:
            f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()

