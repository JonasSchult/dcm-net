
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=1_qem_crops_train
#SBATCH --time=14-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=output/1_qem_crops_train_%A_%a.txt
#SBATCH --array=0-1000

cd ../../../

python3.7 crop_training_samples.py \
--in_path data/scannet_qem_train_rooms/ \
--out_path data/scannet_qem_train_crops/ \
--block_size 3.0 \
--stride 1.5 \
--number $((SLURM_ARRAY_TASK_ID))