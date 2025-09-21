import argparse
import numpy as np
from tqdm import tqdm
from vllm import LLM
import sys
sys.path.append('utils')
from utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages, call_model, get_gemma_chat_prompt
from metrics import metric_max_over_ground_truths, exact_match_score, match, f1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--prompt_name', type=str, default="prompt_input_hop_classification_chat")
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--chat_template', action="store_true")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--compute_pmi', action="store_true")   # this should be False, we dont do pmi in distillation phase

    # sampling params
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="temperature at decoding. Zero means greedy sampling.")                        
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="top-p sampling.")
    parser.add_argument('--top_k', type=int, default=-1,
                    help="top-k sampling.")
    parser.add_argument('--do_stop', action="store_true", default=False)
    parser.add_argument('--logprobs', type=int, default=1,
                    help="number of log probs to return.")       
                            
    
    args = parser.parse_args()

    if args.dtype is not None:
        model = LLM(model=args.model_name, dtype=args.dtype,
                    tensor_parallel_size=args.world_size,) #download_dir=args.download_dir,
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size,) # download_dir=args.download_dir,
    tokenizer = model.get_tokenizer()

    input_data = load_file(args.input_file)
    print('File uploaded, ', args.input_file, len(input_data))

    _prompt_name = args.prompt_name
    for id, item in enumerate(input_data):
        retrieval_result = item["ctxs"]
        question = item["question"]

        #evidences = [retrieval_result[text]["text"] for id, text in enumerate(retrieval_result)]
        #evidences = [retrieval_result["p_0"]["text"]] 
        evidences = ["[1] " + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]

        item["paragraphs"] = evidences

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        # if args.instruction is not None:
        #     item["instruction"] = args.instruction + \
        #         "\n\n### Input:\n" + item["instruction"]
        #     print(item["instruction"] + '\n')

    print(f'Processing data with prompts.')
    processed_batch = []
    for item in input_data:
        for evidence in item["paragraphs"]: # number of evidences will be args.top_n
            item["paragraph"] = evidence
            if args.chat_template and 'gemma' in args.model_name.lower():
                prompt_template = PROMPT_DICT[_prompt_name]
                chat_prompt = get_gemma_chat_prompt(item, prompt_template)
                processed_batch.append(chat_prompt)
            else:
                processed_batch.append(PROMPT_DICT[_prompt_name].format_map(item))
    print(processed_batch[0])

    del item["paragraph"] # just use this dict-key to re-use format
    del item["paragraphs"] # just use this dict-key to re-use format
    del item["instruction"] # just use this dict-key to re-use format

    preds, toklogprobs, _, _, _ = call_model(processed_batch, model, args, tokenizer)
    l = 0
    for j, item in enumerate(input_data):
        for i, ctx in enumerate(item["ctxs"]):
            ctx["hop_pred"] = preds[l]
            l += 1


    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


if __name__ == "__main__":
    main()