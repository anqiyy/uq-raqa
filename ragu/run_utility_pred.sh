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

echo "Training reasoning path utility predictor with musique-hotpotqa-SPLIT.jsonl"

cd /exports/eddie/scratch/s2707044/ragu
export PYTHONPATH=$(pwd)

python passage_utility/main.py \
   --epochs 3 \
   --batch_size 32 \
   --do_train True \
   --save_dir /exports/eddie/scratch/s2707044/ragu/passage_utility/model2 \
   --lr_init 2e-5 \
   --stop_epochs 2 \
   --wd 0.001 \
   --model_name vanilla_bert \
   --input_file /exports/eddie/scratch/s2707044/ragu/data_utility_pred/musique-hotpotqa-SPLIT.jsonl \
   --reference_rank acc_LM \
   --combine_loss be \
   --weight_aux 1 \
   --weight_rank 0 \
   --model_select error \
   --proportion 1 \
   --num_shards 0 \
   --add_title True \

echo "Job Done!"
