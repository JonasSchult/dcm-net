
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=2_qem_rooms_train
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/2_qem_rooms_train_%A_%a.txt
#SBATCH --array=1-200

cd ../../../

python3.7 graph_level_generation.py \
--in_path raw_data/scannet/scans \
--out_path data/scannet_qem_train_rooms/ \
--level_params 0.04 30 30 30 \
--train \
--qem \
--dataset scannet \
--number $((SLURM_ARRAY_TASK_ID + 1000))