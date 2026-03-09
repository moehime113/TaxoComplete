#!/bin/bash
#SBATCH --job-name=taxo-cs-s47
#SBATCH --output=/home/taxo/TaxoComplete/logs/%x-%j.out
#SBATCH --error=/home/taxo/TaxoComplete/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=RTX2080
#SBATCH --gres=gpu:1

set -euo pipefail

cd /home/taxo/TaxoComplete
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load miniconda/3 || true
  module load cuda/11.1 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate taxocomplete

# Use reachable Hugging Face mirror on this cluster.
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_TELEMETRY=1

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
python -c "import torch; print('[INFO] torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python ./src/train.py --config ./config_files/cs/config_clst20_s47.json
