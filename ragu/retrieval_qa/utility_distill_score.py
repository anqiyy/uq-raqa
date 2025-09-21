import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import sys
sys.path.append('utils')
from utils import load_file, PROMPT_DICT, save_file_jsonl, postprocess_answers_closed
from metrics import metric_max_over_ground_truths, exact_match_score, match, f1


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def nliEval(model, tokenizer, premise_raw, hypothesis_raw):
    batch_tokens = tokenizer.batch_encode_plus(list(zip(premise_raw, hypothesis_raw)), padding=True, truncation=True, max_length=512, return_tensors="pt")
    #,truncation_strategy="only_first")
    with torch.no_grad():
        model_outputs = model(**{k: v.to(torch.cuda.current_device()) for k, v in batch_tokens.items()})
    batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
    batch_evids = batch_probs[:, 0].tolist() #entailment_idx
    #batch_conts = batch_probs[:, 1].tolist() #contradiction_idx
    #batch_neuts = batch_probs[:, 2].tolist() #neutral_idx
    #print(batch_evids)
    #exit(0)
    return batch_evids #, batch_neuts, batch_conts    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--nli_model', type=str, default="tals/albert-xlarge-vitaminc-mnli")
    parser.add_argument('--ares_model', type=str, default=None)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--result_fp', type=str)    
    #parser.add_argument('--top_n', type=int, default=5,
    #                    help="number of paragraphs to be considered.")
    
    args = parser.parse_args()

    input_data = load_file(args.input_file)

    if args.nli_model is not None:
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).eval()
        nli_model.to(torch.cuda.current_device())
        nli_model.half()  # use fp16 as in summac

        for item in input_data:
            premise = []
            hypothesis = []
            for i, ctx in enumerate(item["ctxs"]):
            #for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                premise.append(ctx["text"])
                hypothesis.append(item["question"] + " " + ctx["output"])

            entail_results = nliEval(nli_model, nli_tokenizer, premise, hypothesis)
            for i, ctx in enumerate(item["ctxs"]):
                ctx["NLI"] = entail_results[i]

    if args.ares_model is not None:
        print('Not implemented')
        exit(0)

    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


if __name__ == "__main__":
    main()
