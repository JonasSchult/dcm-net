#!/usr/bin/env bash
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=1_qem_rooms_matterport
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/1_qem_rooms_matterport_%A_%a.txt
#SBATCH --array=0-1000

cd ../../../

python3.7 graph_level_generation.py \
--in_path raw_data/matterport/v1/scans \
--out_path data/matterport/ \
--level_params 0.04 30 30 30 \
--train \
--qem \
--dataset matterport \
--number $((SLURM_ARRAY_TASK_ID))