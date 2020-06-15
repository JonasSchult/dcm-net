#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=m_matterport
#SBATCH --signal=TERM@120
#SBATCH --output=eval_matterport_%j.txt

python run.py \
-c experiments/benchmark/matterport/matterport_benchmark.json \
-r model_checkpoints/benchmark/matterport/matterport_benchmark.pth \
-e \
-q
