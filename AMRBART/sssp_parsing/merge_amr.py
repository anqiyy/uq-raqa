# 1. Obtain hop passages Pi: 
# 2. Get AMR graph of each hop passage: 
# 3. Merge AMR graphs: (this script)
# 4. Run SSSP s_path
# 5. Exract Pi from s_path
# python merge_amr.py --input_file /home/s2707044/utilitypred-data/musique-train-hop-combi_llm_subg_AMR.jsonl --output_file data/musique-train-hop-combi_llm_subg_AMR_merged.jsonl


import argparse
import json
import jsonlines
import penman
import networkx as nx
from collections import defaultdict
import re

def sanitize_for_var(s):
    return re.sub(r'[^\w]', '_', s)  # replaces anything not [a-zA-Z0-9_] with '_'


def load_file(input_fp):
    if input_fp.endswith(".json"):
        with open(input_fp, 'r') as f:
            return json.load(f)
    else:
        with jsonlines.open(input_fp, 'r') as reader:
            return list(reader)
        
def save_file_json(data, fp):
    with open(fp, 'w') as f:
        json.dump(data, f, indent=2)  

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def amr_to_networkx(penman_graph):
    G = nx.DiGraph()
    for src, role, tgt in penman_graph.triples:
        if role == ":instance":
            G.add_node(src, label=tgt)
        # else:
        #     G.add_edge(src, tgt, role=role)
        elif not any(c in src for c in ['"', '/']):  # Not a literal or constant
                G.add_node(src)  # Add node without label
        
    # Second pass: Add edges
    for src, role, tgt in penman_graph.triples:
        if role != ":instance":
            G.add_edge(src, tgt, role=role)
    return G


# def merge_amrs_with_disambiguation(amr_strs, align_same_concepts=True):
#     """
#     Merge AMRs with identical node names by:
#     1. Adding graph-specific prefixes to all nodes
#     2. Optionally aligning nodes with identical concepts
#     3. Preserving all original edges
    
#     Args:
#         amr_strs: List of AMR strings
#         align_same_concepts: If True, merge nodes with same concept despite prefixes
#     """
#     # Parse all AMRs and add prefixes
#     prefixed_graphs = []
#     concept_to_vars = defaultdict(list)
    
#     for i, amr in enumerate(amr_strs):
#         g = penman.decode(amr)
#         prefix = f"g{i}_"
#         prefixed_triples = []
        
#         # Track concepts for alignment
#         for src, role, tgt in g.triples:
#             if role == ":instance":
#                 concept_to_vars[tgt].append(f"{prefix}{src}")
        
#         # Add prefixes to all variables (except literals)
#         for src, role, tgt in g.triples:
#             new_src = f"{prefix}{src}" if not src.startswith(('"', '/')) else src
#             new_tgt = f"{prefix}{tgt}" if not tgt.startswith(('"', '/')) else tgt
#             prefixed_triples.append((new_src, role, new_tgt))
        
#         prefixed_graphs.append(penman.Graph(prefixed_triples))
    
#     # Build merged graph
#     merged_triples = []
#     var_mapping = {}  # For concept alignment
    
#     # Step 1: Handle instance triples with optional concept alignment
#     for concept, vars_list in concept_to_vars.items():
#         if align_same_concepts and len(vars_list) > 1:
#             # Align nodes with same concept
#             merged_var = f"m_{concept}"
#             var_mapping.update({v: merged_var for v in vars_list})
#             merged_triples.append((merged_var, ":instance", concept))
#         else:
#             # Keep prefixed nodes separate
#             for var in vars_list:
#                 merged_triples.append((var, ":instance", concept))
    
#     # Step 2: Add all edges with proper variable mapping
#     for g in prefixed_graphs:
#         for src, role, tgt in g.triples:
#             if role != ":instance":
#                 mapped_src = var_mapping.get(src, src)
#                 mapped_tgt = var_mapping.get(tgt, tgt)
#                 merged_triples.append((mapped_src, role, mapped_tgt))
    
#     return penman.encode(penman.Graph(merged_triples))


def merge_amrs_with_disambiguation(amr_strs, align_same_concepts=True):
    prefixed_graphs = []
    concept_literal_to_vars = defaultdict(list)
    var_to_literals = defaultdict(dict)


    def is_variable(x):
        return not (x.startswith('"') or x.replace('.', '', 1).isdigit() or x in {"-", "true", "false"})

    for i, amr in enumerate(amr_strs):
        g = penman.decode(amr)
        prefix = f"g{i}_" # Processing ith hop passage
        prefixed_triples = []

        for src, role, tgt in g.triples:
            # Store literals for each var (e.g., :wiki, :name)
            full_src = f"{prefix}{src}" if is_variable(src) else src
            #print(full_src)
            if role in {":wiki", ":name", ":name-of"}:
                #print(role, tgt)
                var_to_literals[full_src][role] = tgt
        #print("var_to_literals: ", var_to_literals)

        for src, role, tgt in g.triples:
            # new_src = f"{prefix}{src}" if not src.startswith(('"', '/')) else src
            # new_tgt = f"{prefix}{tgt}" if not tgt.startswith(('"', '/')) else tgt

            new_src = f"{prefix}{src}" if is_variable(src) else src
            new_tgt = f"{prefix}{tgt}" if is_variable(tgt) else tgt

            prefixed_triples.append((new_src, role, new_tgt))

            if role == ":instance":
                # Make concept disambiguation key with literals
                wiki = var_to_literals.get(new_src, {}).get(":wiki")
                # print(new_src, wiki)
                wiki_from_var = None if wiki == "-" else wiki

                # wiki_from_var = var_to_literals.get(new_src, {}).get(":wiki") if wik '-' else None
                # print(new_src, wiki_from_var)
                key = (tgt, wiki_from_var)
                concept_literal_to_vars[key].append(new_src)
        
        #print("concept_literal_to_vars: ", concept_literal_to_vars)
        prefixed_graphs.append(penman.Graph(prefixed_triples))

    merged_triples = []
    var_mapping = {}

    # Step 1: Align only nodes with same concept + same :wiki (or no wiki)
    for (concept, wiki), vars_list in concept_literal_to_vars.items():
        if align_same_concepts and len(vars_list) > 1 and wiki:
            if wiki:
                cleaned_wiki = sanitize_for_var(wiki.strip('"'))
                merged_var = f"m_{concept}_{cleaned_wiki}"
            else:
                merged_var = f"m_{concept}"
            # print(merged_var)

            # merged_var = f"m_{concept}_{wiki.strip('\"').replace(' ', '_')}"
            for v in vars_list:
                var_mapping[v] = merged_var
            merged_triples.append((merged_var, ":instance", concept))
        else:
            for v in vars_list:
                merged_triples.append((v, ":instance", concept))
   # print("VAR MAPPING:", var_mapping)
   # print("MERGED TRIPPLES:", merged_triples)

    # Step 2: Add all edges with mapped variables
    for g in prefixed_graphs:
        for src, role, tgt in g.triples:
            if role != ":instance":
                mapped_src = var_mapping.get(src, src)
                mapped_tgt = var_mapping.get(tgt, tgt)
                merged_triples.append((mapped_src, role, mapped_tgt))

    try:
        return penman.encode(penman.Graph(merged_triples))
    except penman.exceptions.LayoutError:
        triples_with_root = add_dummy_root_if_disconnected(merged_triples)
        return penman.encode(penman.Graph(triples_with_root))



def add_dummy_root_if_disconnected(triples):
    # Find all variables used as subjects (sources)
    src_vars = {s for s, r, t in triples if not s.startswith('"')}
    # Find all variables used as targets (excluding literals)
    tgt_vars = {t for s, r, t in triples if not t.startswith('"') and not r == ':instance'}

    # Roots are vars that are never a target — i.e., no incoming edge
    root_vars = src_vars - tgt_vars

    if len(root_vars) > 1:
        dummy_root = "ROOT"
        triples.append((dummy_root, ":instance", "multi-root"))
        for i, rv in enumerate(sorted(root_vars)):
            triples.append((dummy_root, f":subgraph{i}", rv))

    return triples



def main():
    parser = argparse.ArgumentParser(description="Extract SSSP paths from AMR graphs")
    parser.add_argument('--input_file', type=str, required=True, help='Input HotpotQA JSON or JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON or JSONL file')
    parser.add_argument('--no_eg', type=int, required=False)

    args = parser.parse_args()

    hotpotqa_amr = load_file(args.input_file)

    output = []
    
    if not args.no_eg:
        args.no_eg = len(hotpotqa_amr)
    print('DATA LENGTH: ',len(hotpotqa_amr))
    for entry in hotpotqa_amr[:args.no_eg]:
        output_entry = entry
        nx_graphs = []
        lst = []
        
        # for k,v in entry["hop_passages_amr"].items():
            # if "amr" in k:
            #     graph_str = v
            #     penman_graph = penman.decode(graph_str)
            #     lst.append(graph_str)
            #     G = amr_to_networkx(penman_graph)
            #     #print(f"{k}: ", penman.encode(penman_graph))
            #     nx_graphs.append(G)
        for _, ctx in enumerate(entry["ctxs"]):
            if ctx["id"].isdigit() or "+" in ctx["id"]: # do not merge combined passages
                continue
            # Merging AMRs of hop passages
            graph_str = ctx["amr"]
            penman_graph = penman.decode(graph_str)
            lst.append(graph_str)
            G = amr_to_networkx(penman_graph)
            nx_graphs.append(G)

        merged = merge_amrs_with_disambiguation(lst, align_same_concepts=True)

        output_entry["merged_graph_disamb"] = merged
        output.append(output_entry)

    # Save output
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Saved to {args.output_file} with size {len(hotpotqa_amr)}")


if __name__ == "__main__":
    main()

