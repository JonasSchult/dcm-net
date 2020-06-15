
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_euc_geo_knn_qem

#SBATCH --signal=TERM@120
#SBATCH --output=eval_euc_geo_knn_qem_%A_%a.txt

#SBATCH --array=0-10



python run.py \
-c experiments/ablation_study/dual_space/euclidean_geodesic/dual_geodesic_euclidean_knn_qem.json \
-r model_checkpoints/ablation_study/dual_space/euclidean_geodesic/dual_geodesic_euclidean_knn_qem.pth \
-e \
-q
