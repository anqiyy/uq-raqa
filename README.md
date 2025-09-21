# Thesis Code and Data – Utility Predictors for RAQA

This repository contains the code used in my MSc thesis to investigate **uncertainty quantification (UQ)** in retrieval-augmented multi-hop question answering (RAQA).
It includes scripts for preprocessing, AMR parsing, training the **Graph Utility Predictor**.

```graphQL 
PROJECT/
│
├── AMRBART/                  # Code and scripts for AMR parsing with AMRBART
│   ├── examples/              # Example usage and configs
│   ├── fine-tune/             # Fine-tuning configs and scripts
│   ├── pre-train/             # Pretraining configs and scripts
│   ├── root/                  # Root directory from original AMRBART release
│   ├── amrbart.py             # Main parser script
│   ├── bart_eddie.sh          # Cluster submission script for parsing
│   └── requirements.yml       # Environment file for AMRBART
│
├── ragu/                     # Main codebase for RAQA uncertainty experiments
│   ├── code/                  # Core model and predictor code
│   ├── data_creation/         # Scripts for building training data
│   ├── data_utility_pred/     # Data used in training/evaluating utility predictors
│   ├── passage_utility/       # Passage Utility Predictor implementation
│   ├── retrieval_qa/          # Scripts for retrieval-augmented QA
│   ├── semantic_uncertainty/  # Entropy-based and semantic uncertainty baselines
│   ├── utils/                 # Utility functions shared across modules
│   ├── combine_data.py        # Script to merge datasets for training
│   ├── run_acc_nli.sh         # Run script for NLI accuracy baseline
│   ├── run_classifier.sh      # Train/evaluate hop classifier
│   ├── run_gemma3.sh          # Run Gemma-3/4B models for QA/subgraph extraction
│   ├── run_utility_llm.sh     # LLM-based utility 
│   ├── run_utility_pred.sh    # Graph Utility training script
│   └── test_utility_pred.sh   # Testing Graph utility predictor
│
├── scripts/                   # Other scripts used for plotting, evaluation and augmenting datasets
│
├── amrbart_final.yml          # yml file to create environment for AMRBART
├── gemma4b.yml                # yml file to create environment for annotations, training and testing Graph Utility Predictor
│
└── README.md

```

---


## Code origins

* The `AMRBART/` directory is based on [goodbai-nlp/AMRBART](https://github.com/goodbai-nlp/AMRBART), adapted for AMR parsing of retrieved passages in this thesis.
* The `ragu/` directory is based on [lauhaide/ragu](https://github.com/lauhaide/ragu/tree/main), extended with code for Graph Utility predictor 


## Datasets

In this thesis, we used two benchmark multi-hop QA datasets:

* **HotpotQA** (Yang et al., 2018)
  Download from: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)

* **MuSiQue** (Trivedi et al., 2022)
  Download from: [https://github.com/stonybrooknlp/musique](https://github.com/stonybrooknlp/musique)

### Passage Retrieval

To obtain candidate passages, we use **Contriever (MS MARCO fine-tuned)** as retriever.

This produces the top-5 retrieved passages per question. 

---

## Environments Needed

Two main environments are required:

1. **amrbart\_final.yml** → creates an environment named `amrbart`
   (for AMR parsing with AMRBART)

2. **gemma4b.yml** → creates an environment named `gemma4b`
   (for QA experiments with Gemma-3/4B and SSSP graph features)

Create and activate environments:

```bash
conda env create -f amrbart_final.yml
conda env create -f gemma4b.yml

conda activate amrbart   # for AMR parsing
conda activate gemma4b   # for Gemma/SSSP and predictor training
```

---

## Training Data Construction

### 1. AMR Parsing

Run AMR parsing on passages with AMRBART (to be downloaded).
On the Eddie cluster (example):

```bash
qsub bart_eddie.sh
```

This script calls `amrbart.py`, giving input file, output file paths and model path to AMRBART. 

### 2. Subgraph Extraction (SSSP)

From parsed AMRs, construct shortest single-source paths (SSSP) for candidate graphs using the Gemma-3/4B environment.
Scripts for extracting subgraphs are in `AMRBART/sssp_parsing`.

1. Run `merge_amr.py`:
```bash
python merge_amr.py --input_file /utilitypred-data/musique-train-hop-combi_llm_subg_AMR.jsonl --output_file data/musique-train-hop-combi_llm_subg_AMR_merged.jsonl
```

2. Run `sssp.py`:
```bash
python sssp.py --input_pg_file data/hotpot-dev-hop-non-combi_AMR_merged.jsonl --output_file data/hotpot-dev-hop-non-combi_AMR_merged_sssp.jsonl --fields "merged_graph_disamb" 
```

3. Run `extract_subg_from_sssp.py` to obtain final SSSP used in training data:
```bash
python extract_subg_from_sssp.py --input_file data/hotpot-dev-mixed-half-golds_ctREAR3-score-qwen2-72b-it-awq-TIT_AMR_merged_sssp.jsonl --output_file data/hotpot-dev-mixed-half-golds_ctREAR3-score-qwen2-72b-it-awq-TIT_AMR_merged_sssp_subg.jsonl
```

We also obtain subgraphs using Gemma3-4b directly.

For this, use `gemmaa4b` environment. 
Run `ragu/run_gemma3.sh` bash script. 

### 3. Annotations (`acc_LM` labels)

During preprocessing, each passage (or passage set) is annotated with `acc_LM ∈ {0,1}`, denoting whether the QA model answered the question correctly given that input.
These labels are used as supervision for both utility predictors.
1. Run `ragu/run_utility_distill_llm.sh`
2. Run `ragu/run_acc_nli.sh`

---

## Training

### Train the Graph Utility Predictor

Run `ragu/run_utility_pred.sh`


### Test the Graph Utility Predictor

Run `ragu/test_utility_pred.sh`

---

## Notes for Future Work

* In `utils/utils.py`:
we attempted to use Gemma3-4b for hop passage classification using the prompt "prompt_input_hop_classification_chat"

-


