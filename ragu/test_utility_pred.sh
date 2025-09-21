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

FILE_NAME="data_utility_pred/musique-combi-test_ctREAR3_annotated.jsonl"

cd /exports/eddie/scratch/s2707044/ragu
export PYTHONPATH=$(pwd)
echo "Running reasoning path utility predictor $FILE_NAME"


python passage_utility/main.py \
   --test_mode test \
   --do_test True \
   --save_dir passage_utility/model2 \
   --model_name vanilla_bert \
   --input_file $FILE_NAME \
   --output_pred_utilities True \
   --model_select error \
   --reference_rank acc_LM \
   --combine_loss be \
   --weight_aux 1 \
   --weight_rank 0 \
   --top_n 5 \
   --add_title True



echo "Job Done!"

