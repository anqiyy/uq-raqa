#!/bin/bash
# Grid Engine options:

#$ -cwd                    # Run from current directory
#$ -q gpu                  
#$ -l gpu=1                
#$ -pe sharedmem 3         
#$ -l h_vmem=64G            
#$ -o logs/amrbart_gpu_train12.out      
#$ -e logs/amrbart_gpu_train12.err     

# Load modules and activate conda
echo "Loading environment..."
. /etc/profile.d/modules.sh

module load anaconda/2024.02  
source activate /exports/eddie/scratch/$USER/amrbart

# Set OpenMP threads to avoid oversubscription
export OMP_NUM_THREADS=1

FILE_NAME="/exports/eddie/scratch/s2707044/ragu/data/hotpot-dev-hop-non-combi_ctREAR3.jsonl"
OUTPUT_FILE="/exports/eddie/scratch/s2707044/ragu/data/train_hop_chunk_id/hotpot-dev-hop-non-combi_amr_full_ctREAR3.jsonl"
MODEL_PATH="/exports/eddie/scratch/$USER/amrbart_model"

echo "starting now"
echo "Running AMRBART on $FILE_NAME"

python amrbart.py \
  --file_input $FILE_NAME \
  --file_output $OUTPUT_FILE \
  --model_path $MODEL_PATH \
  --batch_size 10 \
  --fields ctxs questions


echo "Job is done!"
