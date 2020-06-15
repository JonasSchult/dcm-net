#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_leaky_relu
#SBATCH --signal=TERM@120
#SBATCH --output=eval_leaky_relu_%A_%a.txt
#SBATCH --array=0-10

python run.py \
-c experiments/supplementary/activation_function/leaky_relu.json \
-r model_checkpoints/supplementary/activation_function/leaky_relu.pth \
-e \
-q
