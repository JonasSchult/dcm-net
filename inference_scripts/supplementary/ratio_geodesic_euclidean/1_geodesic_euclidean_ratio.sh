#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_1_geo_euc_ratio
#SBATCH --signal=TERM@120
#SBATCH --output=eval_1_geo_euc_ratio_%A_%a.txt
#SBATCH --array=0-10

python run.py \
-c experiments/supplementary/ratio_geodesic_euclidean/1_geodesic_euclidean_ratio.json \
-r model_checkpoints/supplementary/ratio_geodesic_euclidean/1_geodesic_euclidean_ratio.pth \
-e \
-q
