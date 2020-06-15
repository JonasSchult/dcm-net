
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_radius_fps

#SBATCH --signal=TERM@120
#SBATCH --output=eval_radius_fps_%A_%a.txt

#SBATCH --array=0-10



python run.py \
-c experiments/ablation_study/single_space/single_euclidean/single_euclidean_radius_fpsp_qem.json \
-r model_checkpoints/ablation_study/single_space/single_euclidean/single_euclidean_radius_fpsp.pth \
-e \
-q
