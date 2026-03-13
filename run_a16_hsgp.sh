#!/bin/bash
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=hsgp_online_a16_%j.out

cd /vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP
source .venv/bin/activate

/usr/bin/nvidia-smi

python -m scripts.onedim.solar.test_hsgp_online_fixedkernel \
  --data solar --kernel_type se \
  --seed_num 0 --split_replace 1 \
  --inducing_size 100 --rff_sample_size 1000 --rff_sample_size_qb 5000 \
  --rff_sample_size_pred 5000 \
  --num_iters 5000 \
  --step_size 2.45 \
  --model_full_dir ./models --fullgp_ckpt steps_nll2000 \
  --out_dir ./ohsgp/solar_dataset/models0
