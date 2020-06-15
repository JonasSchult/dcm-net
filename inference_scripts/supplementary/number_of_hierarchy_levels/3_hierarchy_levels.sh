#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --job-name=e_3_hierarchy
#SBATCH --signal=TERM@120
#SBATCH --output=eval_3_hierarchy_%A_%a.txt
#SBATCH --array=0-10

python run.py \
-c experiments/supplementary/number_of_hierarchy_levels/3_hierarchy_levels.json \
-r model_checkpoints/supplementary/number_of_hierarchy_levels/3_hierarchy_levels.pth \
-e \
-q
