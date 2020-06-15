#!/usr/bin/env bash
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=3_qem_crops
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/3_qem_crops_%A_%a.txt
#SBATCH --array=1-193

cd ../../../

python3.7 crop_training_samples.py \
--in_path data/matterport/ \
--out_path data/matterport_crops/ \
--block_size 2.0 \
--stride 1.0 \
--number $((SLURM_ARRAY_TASK_ID + 2000))