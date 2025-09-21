#!/bin/bash
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -pe sharedmem 2
#$ -l h_vmem=64G
#$ -o logs/llm.out
#$ -e logs/llm.err

. /etc/profile.d/modules.sh
module load anaconda/2024.02

module load cuda/11.8

source activate /exports/eddie/scratch/s2707044/gemma4b
echo "Environment activated"

#FILE_NAME="/exports/eddie/scratch/s2707044/ragu/data_utility_pred/hotpot-dev-mixed-all-golds_ctREAR3-score-qwen2-72b-it-awq-TIT_AMR_merged_sssp_subg"
FILE_NAME="data_utility_pred/hotpot-dev-mixed-half-golds-annotated"
echo "Running gemma3 classifier on $FILE_NAME"

python retrieval_qa/gemma3_classify_hops.py \
    --model_name /exports/eddie/scratch/s2707044/gemma-3-4b-it \
    --input_file "${FILE_NAME}.jsonl" \
    --result_fp "${FILE_NAME}-hop_pred.jsonl" \
    --prompt_name "prompt_input_hop_classification_chat" \
    --chat_template \
    --max_new_tokens 50 \
    --do_stop


echo "Done! File saved to ${FILE_NAME}-hop_pred.jsonl"

