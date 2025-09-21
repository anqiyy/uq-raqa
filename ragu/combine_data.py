import random
import jsonlines
import json
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

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# combine with training 
def combine_jsonl(file1, file2, output_file):
    lines = []

    # Read all lines from both files
    with open(file1, "r", encoding="utf-8") as f1:
        lines.extend([line.strip() for line in f1 if line.strip()])

    with open(file2, "r", encoding="utf-8") as f2:
        lines.extend([line.strip() for line in f2 if line.strip()])

    # Shuffle them randomly
    random.shuffle(lines)

    # Write back to output
    with open(output_file, "w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + "\n")
def combine_jsonl1(file1, file2, output_file, max_file1=4):
    lines = []
    for entry in load_file(file1):
        entry["ctxs"] = entry["ctxs"][:4]
        lines.append(entry)
    # Read only first `max_file1` lines from file1
    #with open(file1, "r", encoding="utf-8") as f1:
        #file1_lines = [line.strip() for line in f1 if line.strip()]
        #lines.extend(file1_lines[:max_file1])
    data1 = load_file(file2)
    combined = lines + data1
    random.shuffle(combined)
    write_jsonl(output_file, combined)
    # Read all lines from file2
    #with open(file2, "r", encoding="utf-8") as f2:
    #    lines.extend([line.strip() for line in f2 if line.strip()])

    # Shuffle them randomly
    #random.shuffle(lines)

    # Write back to output
    #with open(output_file, "w", encoding="utf-8") as fout:
    #    for line in lines:
    #        fout.write(line + "\n")


path1 = "data_utility_pred/musique-combi-dev_ctREAR3_annotated_llm_subg_fortrain.jsonl"
path2 = "data_utility_pred/hotpot-dev-hop-non-combi-annotated-sssp-subg.jsonl"
output = "data_utility_pred/musique-hotpotqa-dev.jsonl"
combine_jsonl1(path2, path1, output)
print("done, saved ", output)
