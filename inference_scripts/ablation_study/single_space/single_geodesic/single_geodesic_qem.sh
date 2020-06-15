
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_geo_qem

#SBATCH --signal=TERM@120
#SBATCH --output=eval_geo_qem_%A_%a.txt

#SBATCH --array=0-10



python run.py \
-c experiments/ablation_study/single_space/single_geodesic/single_geodesic_qem.json \
-r model_checkpoints/ablation_study/single_space/single_geodesic/single_geodesic_qem.pth \
-e \
-q
