# python sssp.py --input_pg_file data/hotpot-dev-hop-non-combi_AMR_merged.jsonl --output_file data/hotpot-dev-hop-non-combi_AMR_merged_sssp.jsonl --fields "merged_graph_disamb" 
# "merged_graph"
import argparse
import json
import jsonlines
import penman
import networkx as nx
from typing import List
from collections import defaultdict

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

def is_int(s):
    if s:
        try:
            int(s)
            return True
        except ValueError:
            return False
    return False


def amr_to_networkx(penman_graph):

    G = nx.DiGraph()

    # First pass: add nodes for :instance
    for src, role, tgt in penman_graph.triples:
        if role == ':instance':
            # print("first :instance role: ", src)
            G.add_node(src, label=tgt)
        elif not any(c in src for c in ['"', '/']):  # src is not a literal
            # print("first: ", src)
            G.add_node(src)  # unlabeled structural node (e.g., m_book_-)

    # Second pass: add edges and handle literals
    for src, role, tgt in penman_graph.triples:
        # print(src, tgt, role)
        if role != ':instance':
            G.add_edge(src, tgt, role=role)

            # Check if tgt is a quoted literal
            if isinstance(tgt, str) and tgt.startswith('"') and tgt.endswith('"'):
                node_data = G.nodes[tgt]
                if 'label' not in node_data and 'value' not in node_data:
                    G.nodes[tgt]['label'] = 'UNK'
                    G.nodes[tgt]['value'] = tgt.replace('_', ' ')
            elif is_int(tgt):
                label = 'temporal-quantity'  # default
                if role == ':year':
                    label = 'year'
                elif role == ':month':
                    label = 'month'
                elif role == ':day':
                    label = 'day'
                G.nodes[tgt]['label'] = label
                G.nodes[tgt]['value'] = str(tgt)
    return G


def is_sublist(sub, full):
    """Returns True if `sub` is a contiguous sublist of `full`."""
    if len(sub) >= len(full):
        return False
    for i in range(len(full) - len(sub) + 1):
        if full[i:i + len(sub)] == sub:
            return True
    return False


def filter_maximal_paths(paths):
    """Removes paths that are contiguous subpaths of others."""
    maximal_paths = []
    for path in paths:
        if not any(is_sublist(path, other) for other in paths if path != other):
            maximal_paths.append(path)
    return maximal_paths


def extract_paths_from_graph_qn(graph_str):
    """
    Runs singe_source_shortest_paths from "question-01" node
    """
    penman_graph = penman.decode(graph_str)
    G = amr_to_networkx(penman_graph)

    # Find the source node (question-01)
    source_node = None
    for node, attr in G.nodes(data=True):
        if attr.get("label") == "question-01":
            source_node = node
            break

    if source_node is None:
        print("No 'question-01' node found.")
        return []

    # Get ALL shortest paths from source_node to all reachable nodes
    all_shortest_paths = nx.single_source_all_shortest_paths(G, source=source_node)
    paths_dict = dict(all_shortest_paths)

    
    # Flatten the dictionary of paths into a list
    paths = []
    for target in paths_dict:
        # Skip the trivial path (just the source node itself)
        if target == source_node:
            continue
        # Add all shortest paths to this target
        # paths.extend(all_shortest_paths[target])
        for path in paths_dict[target]:
            paths.append(path)
    
    return filter_maximal_paths(paths)

def extract_question_entities(amr_str):
    graph = penman.decode(amr_str)
    G = nx.DiGraph()

    # Build the graph and collect labels and wiki
    node_labels = {}
    node_wiki = {}
    
    for src, rel, tgt in graph.triples:
        if rel == ":instance":
            node_labels[src] = tgt
        elif rel == ":wiki":
            node_wiki[src] = tgt.strip('"')  # Remove quotes
        elif rel != ":top":
            G.add_edge(src, tgt, role=rel)
    # print(G)
    G = amr_to_networkx(graph)
    # print(G)
    # Step 1: find the root question node (usually the one with label 'question-01')
    source = None
    for node, label in node_labels.items():
        if label == "question-01":
            source = node
            break
    if not source:
        return []

    # Step 2: BFS from question-01 to get all reachable nodes
    reachable = nx.descendants(G, source)

    # Step 3: Collect nodes that have either a :wiki or :name-of/:name and are reachable
    entities = set()
    for node in reachable:
        if node in node_wiki:
            entities.add(node_wiki[node])
        elif node_labels.get(node) not in {"question-01", "amr-unknown", "thing"}:
            # Accept named entities with name-of or name children
            entities.add(node_labels.get(node))

    return sorted(entities)

def extract_question_concepts_from_amr_graph(graph_str):
    graph_str_1 = penman.decode(graph_str)
    graph = amr_to_networkx(graph_str_1)
    # Use root or top concept as starting point
    root = None
    for node, attr in graph.nodes(data=True):
        if attr.get("label") and not attr["label"].startswith('"'):
            root = node
            break
    if root is None:
        return []

    # DFS to find meaningful concepts (skip amr-unknown etc.)
    visited = set()
    concepts = set()

    def dfs(v):
        if v in visited:
            return
        visited.add(v)
        label = graph.nodes[v].get("label", "")
        if label and label not in {"amr-unknown"}:
            concepts.add(label)
        for _, neighbor in graph.out_edges(v):
            dfs(neighbor)

    dfs(root)
    return list(concepts)

def extract_meaningful_nodes(graph_str, include_literals=True):
    graph = penman.decode(graph_str)
    concepts = set()
    roles = set()
    literals = set()

    for src, role, tgt in graph.triples:
        if role == ':instance':
            concepts.add(tgt)
        else:
            roles.add(role.lstrip(':'))
            if include_literals and (tgt.startswith('"') or tgt.startswith("'")):
                literals.add(tgt.strip('"'))

    return {
        "concepts": sorted(concepts),
        "roles": sorted(roles),
        "literals": sorted(literals) if include_literals else []
    }


def find_matching_nodes(concepts, merged_graph):
    matches = set()
    for node, data in merged_graph.nodes(data=True):
        # print(node, data)
        label = data.get("label", "")
        if label in concepts:
            matches.add(node)
    return matches

def run_sssp_from_matches(merged_graph, source_nodes, max_depth=6):
    all_paths = []
    for source in source_nodes:
        paths = nx.single_source_shortest_path(merged_graph, source, cutoff=max_depth)
        for target, path in paths.items():
            if source != target:
                all_paths.append(path)
    return all_paths


def get_node_literal_name(G, node):
    name_parts = []

    # Check if the node itself is a name node with :opN
    node_data = G.nodes.get(node, {})
    if node_data.get('label') == 'name':
        op_edges = sorted(
            [(r, tgt) for _, tgt, r in G.out_edges(node, data=True)
             if r.get('role', '').startswith(':op')],
            key=lambda x: int(x[0]['role'][3:]) if x[0]['role'][3:].isdigit() else 0
        )
        for role_data, literal_node in op_edges:
            val = _extract_literal_value(G, literal_node)
            if val:
                name_parts.append(val)
        if name_parts:
            return " ".join(name_parts)

    # Case 1: :wiki (use the target of :wiki if it has usable data)
    for _, wiki_target, data in G.out_edges(node, data=True):
        if data.get('role') == ':wiki':
            wiki_data = G.nodes.get(wiki_target, {})
            for key in ['value', 'label', 'instance', 'name']:
                if key in wiki_data and isinstance(wiki_data[key], str):
                    return wiki_data[key].strip('"')
            return wiki_target.strip('"')  # fallback to the literal ID

    # Case 2: :name → :op1, :op2, :op3 ...
    for _, name_node, edge_data in G.out_edges(node, data=True):
        if edge_data.get('role') in {':name', ':name-of'}:
            # print("getting literal name node:", name_node, edge_data)
            op_edges = sorted(
                [(r, tgt) for _, tgt, r in G.out_edges(name_node, data=True) if r.get('role', '').startswith(':op')],
                key=lambda x: int(x[0]['role'][3:]) if x[0]['role'][3:].isdigit() else 0
            )
            for role_data, literal_node in op_edges:
                literal_data = G.nodes.get(literal_node, {})
                val = None
                for key in ['value', 'label', 'instance', 'name']:
                    if key in literal_data and isinstance(literal_data[key], str):
                        val = literal_data[key].strip('"')
                        break
                if not val:
                    val = literal_node.strip('"')  # fallback: literal node ID
                if val:
                    name_parts.append(val)

    if name_parts:
        return " ".join(name_parts)

    # Case 3: fallback to direct attributes of the node
    node_data = G.nodes.get(node, {})
    for key in ['value', 'label', 'instance', 'name']:
        val = node_data.get(key)
        if isinstance(val, str):
            return val.strip('"')

    # Final fallback: return node ID if it's a string literal
    if isinstance(node, str):
        return node.strip('"')

    return ""


def _extract_literal_value(G, node):
    node_data = G.nodes.get(node, {})
    for key in ['value', 'label', 'instance', 'name']:
        if key in node_data and isinstance(node_data[key], str):
            return node_data[key].strip('"')
    if isinstance(node, str):
        return node.strip('"')
    return None

def get_path_concepts(path, merged_graph, stop_concepts=None):
    if stop_concepts is None:
        stop_concepts = set()

    concepts = []

    for node in path:
        # Check if node is a quoted literal
        if isinstance(node, str) and node.startswith('"') and node.endswith('"'):
            label = merged_graph.nodes[node].get('value','').strip().lower()
        else:
            label = merged_graph.nodes[node].get('label', '').strip().lower()
        # literal = get_node_literal_name(merged_graph, node).strip().lower()
        # print(literal)

        # Only keep non-empty, non-structural labels/literals
        if label and label not in stop_concepts:
            concepts.append(label)
        # if literal and literal not in stop_concepts:
        #     concepts.append(literal)

    # Remove empty and duplicates while preserving order
    seen = set()
    clean_concepts = []
    for c in concepts:
        if c and c not in seen:
            seen.add(c)
            clean_concepts.append(c)
    
    return clean_concepts



def clean_path_concepts(concepts: List[str], question_concepts=None) -> List[str]:
    structure_fillers = {
        "name", "person", "thing", "entity", "book", "value", "ordinal-entity"
    }
    stop_literals = {"the", "a", "an"}  # you can expand this list
    
    cleaned = []
    for c in concepts:
        c_lower = c.lower()
        if c_lower in structure_fillers:
            # Keep if it's part of the question concepts (optional)
            if question_concepts and c_lower in question_concepts:
                cleaned.append(c)
            continue
        if c.startswith('"') and c.endswith('"'):
            lit = c[1:-1].lower()
            if lit in stop_literals:
                continue
            cleaned.append(c)
        else:
            cleaned.append(c)
    
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped

def group_by_path_prefix(paths, strip_from_end=1):
    """
    Group paths by shared prefix (excluding the final N nodes),
    and consolidate leaves that share the same first leaf node.
    """
    prefix_to_raw_leaves = defaultdict(list)

    for path in paths:
        if len(path) <= strip_from_end:
            prefix_to_raw_leaves[tuple(path[:-1])].append((path[-1],))
            continue

        prefix = tuple(path[:-strip_from_end])
        leaf = tuple(path[-strip_from_end:])
        prefix_to_raw_leaves[prefix].append(leaf)

    # Consolidate leaves that share the same first leaf node
    prefix_to_consolidated = {}

    for prefix, raw_leaves in prefix_to_raw_leaves.items():
        grouped_leaves = defaultdict(list)
        for leaf in raw_leaves:
            root = leaf[0]
            grouped_leaves[root].append(leaf)

        consolidated_leaves = []
        for root, leaf_group in grouped_leaves.items():
            # Flatten the grouped leaves while preserving order
            flattened = [item for leaf in leaf_group for item in leaf[1:]]
            consolidated_leaves.append((root, *flattened))

        prefix_to_consolidated[prefix] = consolidated_leaves

    return prefix_to_consolidated


def flatten(x):
    result = []
    for item in x:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result



def main():
    parser = argparse.ArgumentParser(description="Extract SSSP paths from AMR graphs")
    parser.add_argument('--input_pg_file', type=str, required=True, help='Input JSON or JSONL file')
    # parser.add_argument('--input_qn_file', type=str, required=True, help='Input JSON or JSONL file')
    parser.add_argument('--fields', nargs='+', required=True, help="field for SSSP") # "merged_graph_disamb"
    parser.add_argument('--output_file', type=str, required = True, help = "Output JSONL file")
    args = parser.parse_args()

    data = load_file(args.input_pg_file)
    # data_qn = load_file(args.input_qn_file)

    output = []
    for entry in data:
        #print("---------------------------q_id: ", d["q_id"], " -----------------------------------------------------------")
        
        output_data = entry
        qn_amr = entry.get("question_amr", "")
        result = extract_meaningful_nodes(qn_amr)
        question_concepts = result['concepts'] + result['literals']
        
        # Run sssp for non-hops to question
        for i,ctx in enumerate(entry["ctxs"]):
            # print("Looking at ", ctx)
            if ctx["id"].isdigit() or "+" in ctx["id"]:
                graph_str = ctx.get("sssp_output", "")
                # print(graph_str)
                graph_str = penman.decode(graph_str)
                graph = amr_to_networkx(graph_str)
                matches = find_matching_nodes(question_concepts, graph)
                if len(matches) > 0:
                    final_path = filter_maximal_paths(run_sssp_from_matches(graph, matches))
                    nodes = []
                    for path in final_path:
                        nodes_in_path = get_path_concepts(path, graph)
                        cleaned = clean_path_concepts(nodes_in_path, question_concepts)
                        nodes.append(cleaned)
                    grouped = group_by_path_prefix(final_path, 2)

                    labelled_paths = []
                    merged_node_paths = []

                    for prefix, to_merge in grouped.items():
                        #print(prefix, to_merge)
                        merged_nodes = flatten([prefix, to_merge])
                        merged_node_paths.append(merged_nodes)
                        
                        nodes_labels = [
                            graph.nodes[node].get('value', 'UNK')
                                if graph.nodes[node].get('label') in [None, 'UNK'] or is_int(graph.nodes[node].get('value'))
                                else graph.nodes[node].get('label') 
                                for node in merged_nodes
                        ]
                        # for node in merged_nodes:  
                        #     val = graph.nodes[node].get('value', 'UNK')
                        #     labl = graph.nodes[node].get('label', 'UNK')
                            #print("checks here:", node, val, labl)
                        labelled_paths.append(nodes_labels)


                    output_data["ctxs"][i]["sssp_output"] = labelled_paths
                else:
                    output_data["ctxs"][i]["sssp_output"] = []
            
        # Processing hop passages
        for field in args.fields:
            graph_str = entry.get(field, "")
            print(graph_str)
            if graph_str not in ["()", ""]:
                graph_str = penman.decode(graph_str)
                merged_graph = amr_to_networkx(graph_str)  

                matches = find_matching_nodes(question_concepts, merged_graph)
                #print(matches)
                
                paths = run_sssp_from_matches(merged_graph, matches)
                final_path = filter_maximal_paths(paths)

                nodes = []
                for path in final_path:
                    #print(" -> ".join([
                    #    f"{node} ({merged_graph.nodes[node].get('label', 'UNK')})"
                    #    for node in path
                    #]))
                    nodes_in_path = get_path_concepts(path, merged_graph)
                    cleaned = clean_path_concepts(nodes_in_path, question_concepts)
                    # print(cleaned)
                    nodes.append(cleaned)
                grouped = group_by_path_prefix(final_path, 2)

                merged_labelled_paths = []
                merged_node_paths = []
                for prefix, to_merge in grouped.items():
                    #print(prefix, to_merge)
                    merged_nodes = flatten([prefix, to_merge])
                    merged_node_paths.append(merged_nodes)
                    
                    merged_nodes_labels = [
                        merged_graph.nodes[node].get('value', 'UNK')
                            if merged_graph.nodes[node].get('label') in [None, 'UNK'] or is_int(merged_graph.nodes[node].get('value'))
                            else merged_graph.nodes[node].get('label') 
                            for node in merged_nodes
                    ]
                    for node in merged_nodes:
                        
                        val = merged_graph.nodes[node].get('value', 'UNK')
                        labl = merged_graph.nodes[node].get('label', 'UNK')
                        #print("checks here:", node, val, labl)
                    merged_labelled_paths.append(merged_nodes_labels)

                #print("MERGED:" , merged_labelled_paths)
                # print(merged_node_paths)

                output_data["sssp_raw_"+field] = final_path
                output_data["sssp_cleaned_"+field] = nodes
                output_data["sssp_cleaned_nodes_"+field] = merged_node_paths # with node id
                output_data["sssp_cleaned_label_"+field] = merged_labelled_paths # with labels instead of node id
                output_data["question_concepts"] = result
                # print(nodes)

            output.append(output_data)

    # Save output
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
