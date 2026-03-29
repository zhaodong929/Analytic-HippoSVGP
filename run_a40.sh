#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=a40_%j.out

cd /vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP

source .venv/bin/activate

echo "===== HOSTNAME ====="
hostname
echo "===== DATE ====="
date
echo "===== NVIDIA-SMI ====="
/usr/bin/nvidia-smi

echo "===== TORCH CUDA CHECK ====="
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
