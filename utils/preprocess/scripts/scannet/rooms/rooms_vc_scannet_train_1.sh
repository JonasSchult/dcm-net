
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=1_vc_rooms_train
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/1_vc_rooms_train_%A_%a.txt
#SBATCH --array=0-1000

cd ../../../

python3.7 graph_level_generation.py \
--in_path raw_data/scannet/scans \
--out_path data/scannet_vc_train_rooms/ \
--level_params 0.04 0.08 0.16 0.32 \
--train \
--vertex_clustering \
--dataset scannet \
--number $((SLURM_ARRAY_TASK_ID))
