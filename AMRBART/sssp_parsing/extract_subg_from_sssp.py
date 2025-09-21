# python extract_subg_from_sssp.py --input_file data/hotpot-dev-mixed-half-golds_ctREAR3-score-qwen2-72b-it-awq-TIT_AMR_merged_sssp.jsonl --output_file data/hotpot-dev-mixed-half-golds_ctREAR3-score-qwen2-72b-it-awq-TIT_AMR_merged_sssp_subg.jsonl
import argparse
import json
import jsonlines
import penman
import networkx as nx
from collections import defaultdict
import re


def load_file(input_fp):
    if input_fp.endswith(".json"):
        with open(input_fp, 'r') as f:
            return json.load(f)
    else:
        with jsonlines.open(input_fp, 'r') as reader:
            return list(reader)
        
def is_int(s):
    if s:
        try:
            int(s)
            return True
        except ValueError:
            return False
    return False


def amr_to_networkx(graph_str):
    penman_graph = penman.decode(graph_str)
    G = nx.DiGraph()

    # First pass: add nodes for :instance
    for src, role, tgt in penman_graph.triples:
        if role == ':instance':
            # print("first :instance role: ", src)
            G.add_node(src, label=tgt)
        elif not any(c in src for c in ['"', '/']):  # src is not a literal
            # print("first: ", src)
            G.add_node(src)  # unlabeled structural node (e.g., m_book_-)

    for src, role, tgt in penman_graph.triples:
        # print(src, tgt, role)
        
        if role != ':instance':
            # Ensure src node exists (might be missed in first pass)
            if src not in G:
                G.add_node(src, label='UNK')  # default if missing

            G.add_edge(src, tgt, role=role)

            # Case 1: literal string
            if isinstance(tgt, str) and tgt.startswith('"') and tgt.endswith('"'):
                if tgt not in G:
                    G.add_node(tgt)
                if 'label' not in G.nodes[tgt] and 'value' not in G.nodes[tgt]:
                    G.nodes[tgt]['label'] = 'UNK'
                    G.nodes[tgt]['value'] = tgt.replace('_', ' ')

            # Case 2: integer literal
            elif is_int(tgt):
                if tgt not in G:
                    G.add_node(tgt)
                label = 'temporal-quantity'
                if role == ':year':
                    label = 'year'
                elif role == ':month':
                    label = 'month'
                elif role == ':day':
                    label = 'day'
                G.nodes[tgt]['label'] = label
                G.nodes[tgt]['value'] = str(tgt)

    return G

def get_literal_sources(G, literal_node):
    nodes = [] 
    for src, _ in G.in_edges(literal_node, data=False):
        nodes.append(src)
    return nodes

def main():
    parser = argparse.ArgumentParser(description="Extract SSSP paths from AMR graphs")
    parser.add_argument('--input_file', type=str, required=True, help='Input JSON or JSONL file with SSSP') # 
    parser.add_argument('--fields', nargs='+', required=False)
    parser.add_argument('--output_file', type=str, required = True, help = "Output JSONL file")

    args = parser.parse_args()

    sssp_data = load_file(args.input_file)
    field_with_nodes_id = "sssp_cleaned_nodes_merged_graph_disamb"
    output = []
    for entry in sssp_data:
        output_data = entry

        graph_str = entry.get("merged_graph_disamb", "")
        if graph_str not in ["()", ""]:
            merged_G = amr_to_networkx(entry["merged_graph_disamb"])
            # n_hops = len(entry['hop_passages']) // 2 # {p_0, p1, .. p_0_amr, p_1_amr,.. p_n_amr}
            # Extract nodes related to passages using node label i.e. g0_z8 -> g1
            path_node_extracted = defaultdict(list)
            labelled_node_extracted = defaultdict(list)
            for path in entry[field_with_nodes_id]:
                # ["g0_z8", "\"K._A._Applegate\""] -> extract {0:["g0_z8", "\"K._A._Applegate\""], 1:[]}
                node_hops = []
                for node in path:
                    if isinstance(node, str):
                        match = re.match(r'g(\d+)_', node)
                        if match:
                            hop_idx = int(match.group(1))
                        else:  # literal or shared node
                            src_nodes = get_literal_sources(merged_G, node)
                            try:
                                if len(src_nodes) == 1:
                                    hop_idx = int(re.match(r'g(\d+)_', src_nodes[0]).group(1))
                                else:
                                    hop_idx = None
                            except:
                                hop_idx = None
                        
                        node_hops.append((node, hop_idx)) 

                # Group nodes by hop, keeping original order
                hops_in_path = set(h for _, h in node_hops if h is not None)
                for hop_idx in hops_in_path:
                    subpath = [node for node, h in node_hops if h == hop_idx]
                    path_node_extracted[hop_idx].append(subpath)

            # print(path_node_extracted)
            output_data['path_node_extracted'] = path_node_extracted
            for node_id, paths in path_node_extracted.items():
                labelled_paths = []
                for path in paths:
                    labelled_path = []
                    for node in path:
                        # Keep the node itself if it's a literal (label is None)
                        label = merged_G.nodes[node].get('label')
                        labelled_path.append(merged_G.nodes[node].get('value', "UNK") if label in [None, "UNK"] or is_int(merged_G.nodes[node].get('value')) else merged_G.nodes[node].get('label'))
                    labelled_paths.append(labelled_path)
                labelled_node_extracted[node_id] = labelled_paths
            # print(labelled_node_extracted)
            
            hop_id = 0
            for i, ctx in enumerate(entry["ctxs"]):
                if ctx["id"].isdigit() or "+" in ctx["id"]:
                    continue
                output_data["ctxs"][i]["sssp_output"] = labelled_node_extracted[hop_id]

            # output_data['labelled_node_extracted'] = labelled_node_extracted
            output.append(output_data)
        else:
            output.append(output_data)

    # Save output
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    


if __name__ == "__main__":
    main()

