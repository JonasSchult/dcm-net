#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=m_area5_s3dis
#SBATCH --signal=TERM@120
#SBATCH --output=majority_area5_s3dis_%j.txt

python run.py \
-c experiments/benchmark_inference/s3dis/s3dis_benchmark_area_5.json \
-r model_checkpoints/benchmark/s3dis/s3dis_benchmark_area_5.pth \
-e \
-q
