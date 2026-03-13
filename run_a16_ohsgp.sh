#!/bin/bash
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=ohsgp_analytic_a16_%j.out

cd /vol/bitbucket/zg1425/myproject/Analytic-HippoSVGP
source .venv/bin/activate

/usr/bin/nvidia-smi

python -m scripts.onedim.solar.test_ohsgp_analytic_solar \
  --device cuda \
  --data solar --kernel_type se \
  --seed_num 0 --split_replace 1 \
  --inducing_size 100 --rff_sample_size 1000 --rff_sample_size_qb 5000 \
  --step_size 2.45 \
  --model_full_dir ./models --fullgp_ckpt steps_nll2000 \
  --hyper_source load_fullgp \
  --phase_origin_mode global_start \
  --out_dir ./ohsgp_analytic/solar_dataset_compare/analytic_version
