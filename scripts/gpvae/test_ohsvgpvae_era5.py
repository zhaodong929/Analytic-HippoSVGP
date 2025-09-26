#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HiPPO-SVGP Sequential Learning for Solar Data
This script implements sequential learning tasks using HiPPO-SVGP models.
"""

import random
import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
import json

# Import HiPPO-SVGP components
from hipposvgp.likelihood import GaussianLikelihood
from hipposvgp.kernel import SE_kernel
from hipposvgp.networks import MLPEncoder, MLPDecoder
from hipposvgp.gpvae import HSGPVAEMultipleKernels, HSGPVAEMultipleKernels_EWC
from hipposvgp.hippo import HiPPO_LegS
from scripts.gpvae.data import ERA5Dataset
torch.autograd.set_detect_anomaly(True) 

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
parser = argparse.ArgumentParser(description='HiPPO-SVGP for Solar Data')
parser.add_argument('--inducing_size', type=int, default=10)
parser.add_argument('--step_size', type=float, default=4.65)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
# Model hyperparameters
rff_sample_size = 10
seed = args.seed
inducing_size = args.inducing_size
# step_size = 4.65
# step_size = 3.65
step_size = args.step_size
# step_size = 2.65
# step_size = 1.65
dim_f = 2
dim_y = 17
num_input_axes = 2
lambda_ewc = 0.01

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_and_train_task1(hippo):
    """Create and train the model for Task 1."""

    # Create kernels
    kernel = [
        SE_kernel(1, device=device)
        for _ in range(dim_f)
    ]

    # Create encoder and decoder networks
    encoder = MLPEncoder(dim_y=dim_y, dim_f=dim_f, num_hidden_units=16).to(device)
    decoder = MLPDecoder(dim_f=dim_f, dim_y=dim_y, num_hidden_units=16).to(device)
    

    # Create model for task 1
    #dataset = ERA5Dataset("data/era5/processed_timeseries_2/task_1")
    dataset = ERA5Dataset("data/era5/processed_timeseries_4/task_1")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    start = np.min(np.array([dataset[i][4] for i in range(len(dataset))]))
    end = np.max(np.array([dataset[i][4] for i in range(len(dataset))]))
    maxT = end-start
    num_discrete_steps = math.ceil(maxT / step_size)
    end = start + num_discrete_steps * step_size
    maxT = end - start # 0 - 372

    # Create directory for saving models and plots
    os.makedirs(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task1', exist_ok=True)

    model = HSGPVAEMultipleKernels(
        kernel=kernel,
        likelihood=GaussianLikelihood(0.01),
        encoder=encoder,
        decoder=decoder,
        end=end,
        start=start,
        rff_sample_size=rff_sample_size,
        num_discrete_steps=num_discrete_steps,
        hippo=hippo,
        inducing_size=inducing_size,
        device=device,
        prev_discrete_steps=0,
        num_input_axes=num_input_axes,
        N_train=dataset[0][0].shape[-2],
    ).to(device)
    
    model.load_state_dict(torch.load(f'./svgpvae_{inducing_size}_seed{seed}/task1/model.best.pt'), strict=False)
    #model.likelihood.log_variance.requires_grad = False
    for j in range(model.dim_f):
        model.kernel[j].log_sf.requires_grad = False
        model.kernel[j].log_ls.requires_grad = False

    w = torch.stack(
        [
            model.kernel[i].sample_from_spectral(model.rff_sample_size)
            for i in range(model.dim_f)
        ]
    ).unsqueeze(1).detach().double()
    
    # Train model
    optimizer = Adam(model.parameters(), lr=5e-3)
    num_epochs = 30
    
    best_checkpoint = None  # To store the best checkpoint
    best_val_rmse = float('inf')

    scale = dataset.scale
    scale = torch.tensor(scale, device=device)
    
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in tqdm(enumerate(dataloader)):
            x_batch = time_train.unsqueeze(2).to(device) # [bs, N, 1]
            y_batch = data_train.to(device) # [bs, N, ydim]
            mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
            seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
            optimizer.zero_grad()
            
            #Knns, Kmms, Knms, Kuu_old = model.compute_kernels(x_batch, None)
            if i == 0  and epoch == 0:
                Knns, Kmms, Knms, Kuu_old = model.compute_kernels(x_batch, w=w, recompute_kernel=True)
            else:
                Knns, Kmms, Knms, Kuu_old = model.compute_kernels(x_batch, w=w, recompute_kernel=False)
            loss = -model.elbo(x_batch, y_batch, mask_batch, seq_length_batch, Knns, Kmms, Knms, Kuu_old, \
                               torch.zeros((inducing_size, 1, dim_f, x_batch.shape[0]), device=device),\
                                  torch.zeros((inducing_size, inducing_size, dim_f, x_batch.shape[0]), device=device))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item() * x_batch.shape[0])
        loss_mean = np.sum(np.array(loss_list))/ len(dataset)
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_mean}")

        #Calculate validation RMSE
        with torch.no_grad():
            model.eval()
            # w = torch.stack(
            #     [
            #         model.kernel[i].sample_from_spectral(model.rff_sample_size)
            #         for i in range(model.dim_f)
            #     ]
            # ).unsqueeze(1).detach()

            val_rmse_list = []
            for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in enumerate(dataloader):
                x_train_batch = time_train.unsqueeze(2).to(device)
                y_train_batch = data_train.to(device)
                x_val_batch = time_val.unsqueeze(2).to(device)
                y_val_batch = data_val.to(device)
                mask_batch = torch.ones(x_train_batch.shape[0], x_train_batch.shape[1], 1).to(device).float()
                
                #if i == 0:
                #    Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train_batch, x_val_batch, w=w, recompute_kernel=True)
                #else:
                Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train_batch, x_val_batch, w=w, recompute_kernel=False)
                # print(f"y_train_batch shape: {y_train_batch.shape}, mask_batch shape: {mask_batch.shape}, Kxxs shape: {Kxxs.shape}, Kxms shape: {Kxms.shape}, Kmms shape: {Kmms.shape}, Knms shape: {Knms.shape}, {hasattr(model, 'Z_cache')}")
                pred_y, _, _, _ = model.pred_y_vmap(y_train_batch, mask_batch, 100, Kxxs, Kxms, Kmms, Knms)
                assert pred_y.shape == y_val_batch.shape
                
                # Calculate RMSE with dataset.scale 
                err = (pred_y - y_val_batch)  # [B, T_val, D]
                mse_per_seq = err.pow(2).mean(dim=(1))  # [B, D]
                rmse_per_seq = torch.sqrt(mse_per_seq).mean(dim=(1))  # [B]
                
                # collect each sequence's RMSE
                val_rmse_list.extend(rmse_per_seq.cpu().tolist())
            # Average across all sequences
            val_rmse_mean = float(np.mean(val_rmse_list))
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_mean}, Validation RMSE: {val_rmse_mean}")
            #torch.cuda.empty_cache()
        
        #Save the best checkpoint
        if val_rmse_mean < best_val_rmse:
            best_val_rmse = val_rmse_mean
            best_checkpoint = f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task1/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)
            torch.save(model.state_dict(), best_checkpoint)

    # Load the best model
    model.load_state_dict(torch.load(best_checkpoint))
    best_model_path = f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task1/model.best.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    model.w = w
    
    return model, dataset, end, num_discrete_steps, w


@torch.no_grad()
def batch_params(model, dataset, w):
    """Compute batch parameters for the model using torch.vmap."""
    def _batch_fn(x_train, y_train, mask):
        qnet_mu_old, qnet_var_old = model.encoder(y_train)
        Z_old, _, _ = model.get_Z(w)
        Ku_old_old = Z_old.permute(0,2,1) @ Z_old
        Ku_old_old *= torch.stack([torch.exp(model.kernel[i].log_sf) for i in range(model.dim_f)]).view(-1,1,1) 
        Ku_old_old = torch.stack([model.add_jitter(Ku_old_old[i]) for i in range(model.dim_f)])#, flag_multidim=True) # [dimf, M, M]

        rff_sample_size = w.shape[-1]
        Z_f = (1 / rff_sample_size) ** 0.5 * torch.cat(
            [torch.sin(w * x_train), torch.cos(w * x_train)], -1
        ) # [dimf, N, rff]
        Knms_old = Z_f @ Z_old # [dimf, N, M]
        Knms_old *= torch.stack([torch.exp(model.kernel[i].log_sf) for i in range(model.dim_f)]).view(-1,1,1) 
       
        mu_hat_old, A_hat_old = model.approximate_posterior_inducing_params(
            mask,
            qnet_mu_old,
            qnet_var_old,
            Ku_old_old,
            Knms_old,
        )

        return mu_hat_old, A_hat_old#, Ku_old_old, Z_old
    vmap_fn = torch.vmap(_batch_fn, in_dims=(0, 0, 0), out_dims=(-1, -1))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    res_mu_old = []
    res_A_old = []
    for _, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in enumerate(dataloader):
        x_batch = time_train.unsqueeze(2).to(device) # [bs, N, 1]
        y_batch = data_train.to(device) # [bs, N, ydim]
        mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
        mu_hat_old_batch, A_hat_old_batch = vmap_fn(x_batch, y_batch, mask_batch)
        res_mu_old.append(mu_hat_old_batch)
        res_A_old.append(A_hat_old_batch)
    return torch.concatenate(res_mu_old, dim=-1), torch.concatenate(res_A_old, dim=-1)


def train_next_task(prev_model, prev_data, start, task_idx, prev_discrete_steps, w, hippo):
    """Train model for task_idx (2-10) with EWC."""
    
    # Extract parameters for continual learning
    mu_hat_old, A_hat_old = batch_params(prev_model, prev_data, w=w)
    print(f"mu_hat_old shape: {mu_hat_old.shape}, A_hat_old shape: {A_hat_old.shape}") # [M, 1, dimf, num_of_sequences], [M, M, dimf, num_of_sequences]
    mu_hat_old_np = mu_hat_old.detach().cpu().numpy()
    A_hat_old_np = A_hat_old.detach().cpu().numpy()
    prev_model.mu_hat_old = mu_hat_old
    prev_model.A_hat_old = A_hat_old
    
    # Create directory for saving models and parameters
    os.makedirs(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task{task_idx}', exist_ok=True)
    np.savez(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task{task_idx}/mu_A_old.npz', mu_hat_old=mu_hat_old_np, A_hat_old=A_hat_old_np)

    encoder_old = deepcopy(prev_model.encoder)
    kernel_old = deepcopy(prev_model.kernel)
    
    # Make kernel_old non-trainable
    for k in kernel_old:
        k.log_sf.requires_grad = False
        k.log_ls.requires_grad = False
    
    # Create components for current task
    likelihood_task = deepcopy(prev_model.likelihood)
    likelihood_task.log_variance.requires_grad = False
    encoder_task = deepcopy(prev_model.encoder)
    decoder_task = deepcopy(prev_model.decoder)
    
    # Compute Fisher Information
    # Step 2: Compute Fisher Information & Store Parameters
    optimal_params = {
        name: param.clone().detach()
        for name, param in prev_model.named_parameters()
        if "encoder" in name or "decoder" in name
    }

    # Compute Fisher Information
    prev_dataloader = torch.utils.data.DataLoader(prev_data, batch_size=8, shuffle=False)
    fisher_info = {
        name: 0  # Approximate diagonal Fisher Information
        for name, param in prev_model.named_parameters()
        if param.requires_grad and ("encoder" in name or "decoder" in name)
    }

    print("Computing Fisher Information...")
    for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in tqdm(enumerate(prev_dataloader)): 
        prev_model.zero_grad()
        x_batch = time_train.unsqueeze(2).to(device) # [bs, N, 1]
        y_batch = data_train.to(device) # [bs, N, ydim]
        mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
        seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
        if task_idx > 1:
            Knns, Kmms, Knms, Kuu_old = prev_model.compute_kernels(x_batch, w=w, recompute_kernel=False)
        else:
            if i >= 1:
                Knns, Kmms, Knms, Kuu_old = prev_model.compute_kernels(x_batch, w=w, recompute_kernel=False)
            else:
                Knns, Kmms, Knms, Kuu_old = prev_model.compute_kernels(x_batch, w=w, recompute_kernel=True)
        if task_idx == 1:
            mu_hat_old_batch = torch.zeros((inducing_size, 1, dim_f, x_batch.shape[0]), device=device)
            A_hat_old_batch = torch.zeros((inducing_size, inducing_size, dim_f, x_batch.shape[0]), device=device)
        else:
            mu_hat_old_batch = prev_model.mu_hat_old[..., idx]
            A_hat_old_batch = prev_model.A_hat_old[..., idx]
        if task_idx == 2:
            loss_prev_task = -prev_model.elbo(x_batch, y_batch, mask_batch, seq_length_batch, Knns, Kmms, Knms, Kuu_old, mu_hat_old_batch, A_hat_old_batch) # [bs, 1]
        else:
            loss_prev_task = -prev_model.elbo(x_batch, y_batch, mask_batch, \
                  seq_length_batch, Knns, Kmms, Knms, Kuu_old, mu_hat_old_batch, A_hat_old_batch, add_ewc=False) # [bs, 1]

        for j in range(len(loss_prev_task)):
            loss_prev_task[j].backward(retain_graph=True)
            for name, param in prev_model.named_parameters():
                if param.requires_grad and ("encoder" in name or "decoder" in name):
                    fisher_info[name] += param.grad**2 / len(prev_data)
            prev_model.zero_grad()
        torch.cuda.empty_cache()

    # Prepare data for current task
    #dataset = ERA5Dataset(f"data/era5/processed_timeseries_2/task_{task_idx}")
    dataset = ERA5Dataset(f"data/era5/processed_timeseries_4/task_{task_idx}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Create EWC model for current task
    start_next = start
    end_next = np.max(np.array([dataset[i][4] for i in range(len(dataset))]))
    maxT = end_next-start_next
    num_discrete_steps = math.ceil(maxT / step_size)
    end_next = start_next + num_discrete_steps * step_size
    
    Z_old, _, _ = prev_model.get_Z(w)
    
    Ku_old_old = Z_old.permute(0,2,1) @ Z_old
    Ku_old_old *= torch.stack([torch.exp(prev_model.kernel[i].log_sf) for i in range(prev_model.dim_f)]).view(-1,1,1) 
    Ku_old_old = torch.stack([prev_model.add_jitter(Ku_old_old[i]) for i in range(prev_model.dim_f)])#, flag_multidim=True) # [dimf, M, M]
    
    model_task = HSGPVAEMultipleKernels_EWC(
        kernel=kernel_old,
        likelihood=likelihood_task,
        encoder=encoder_task,
        decoder=decoder_task,
        end=end_next,
        start=start_next,
        rff_sample_size=rff_sample_size,
        num_discrete_steps=num_discrete_steps,
        hippo=hippo,
        inducing_size=inducing_size,
        device=device,
        prev_discrete_steps=prev_discrete_steps,
        num_input_axes=num_input_axes,
        N_train=dataset[0][0].shape[-2],
        mu_hat_old=mu_hat_old, # [M, 1, dimf, num_of_sequences]
        A_hat_old=A_hat_old,   # [M, M, dimf, num_of_sequences]
        encoder_old=encoder_old,
        Kaa_old = Ku_old_old,
        Z_old=Z_old,
        lambda_ewc=lambda_ewc,
    )

    # Load optimal parameters and Fisher Information
    model_task.optimal_params = optimal_params
    model_task.fisher_info = fisher_info
    
    # Train model for current task
    optimizer = Adam(model_task.parameters(), lr=5e-3)
    
    num_epochs = 20
    
    best_checkpoint = None  # To store the best checkpoint
    best_val_rmse = float('inf')

    scale = dataset.scale
    scale = torch.tensor(scale).to(device)
    
    print(f"Training Task {task_idx}")
    for epoch in range(num_epochs):
        model_task.train()
        loss_list = []
        for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in tqdm(enumerate(dataloader)):
            x_batch = time_train.unsqueeze(2).to(device) # [bs, N, 1]
            y_batch = data_train.to(device) # [bs, N, ydim]
            mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
            seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
            
            optimizer.zero_grad()
            if i == 0 and epoch == 0:
                Knns, Kmms, Knms, Kuu_old = model_task.compute_kernels(x_batch, w=w, recompute_kernel=True)
            else:
                Knns, Kmms, Knms, Kuu_old = model_task.compute_kernels(x_batch, w=w, recompute_kernel=False)
            mu_hat_old_batch = model_task.mu_hat_old[..., idx]
            A_hat_old_batch = model_task.A_hat_old[..., idx]
            loss = -model_task.elbo(x_batch, y_batch, mask_batch, seq_length_batch, Knns, Kmms, Knms, Kuu_old, mu_hat_old_batch, A_hat_old_batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()* x_batch.shape[0])
        loss_mean = np.sum(np.array(loss_list))/ len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_mean}")

        #Calculate validation RMSE
        model_task.eval()
        with torch.no_grad():
            val_rmse_list = []
            for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in enumerate(dataloader):
                x_train_batch = time_train.unsqueeze(2).to(device)
                y_train_batch = data_train.to(device)
                x_val_batch = time_val.unsqueeze(2).to(device)
                y_val_batch = data_val.to(device)
                mask_batch = torch.ones(x_train_batch.shape[0], x_train_batch.shape[1], 1).to(device).float()
                
                if hasattr(model_task, 'Z_cache'):
                    Kxxs, Kxms, Kmms, Knms = model_task.compute_kernels_val(x_train_batch, x_val_batch, w=w, recompute_kernel=False)
                else:
                    Kxxs, Kxms, Kmms, Knms = model_task.compute_kernels_val(x_train_batch, x_val_batch, w=w, recompute_kernel=True)
                pred_y, _, _, _ = model_task.pred_y_vmap(y_train_batch, mask_batch, 5, Kxxs, Kxms, Kmms, Knms)
                assert pred_y.shape == y_val_batch.shape
                
                # Calculate RMSE with dataset.scale 
                err = (pred_y - y_val_batch)  # [B, T_val, D]
                mse_per_seq = err.pow(2).mean(dim=(1))  # [B, D]
                rmse_per_seq = torch.sqrt(mse_per_seq).mean(dim=(1))  # [B]
                
                # collect each sequence's RMSE
                val_rmse_list.extend(rmse_per_seq.cpu().tolist())
                
            # Average across all sequences
            val_rmse_mean = float(np.mean(val_rmse_list))
            print(f"Epoch {epoch+1}/{num_epochs}, Validation RMSE: {val_rmse_mean}")
            torch.cuda.empty_cache()
        
        # Save the best checkpoint
        if val_rmse_mean < best_val_rmse:
            best_val_rmse = val_rmse_mean
            best_checkpoint = f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task{task_idx}/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)
            torch.save(model_task.state_dict(), best_checkpoint)
    
    #Load the best model
    model_task.load_state_dict(torch.load(best_checkpoint))
    best_model_path = f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/task{task_idx}/model.best.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model_task.state_dict(), best_model_path)
    
    # Set the HiPPO weights
    model_task.w = w
    
    return model_task, dataset, num_discrete_steps, end_next


@torch.no_grad()
def evaluate_test_metrics(model, current_dataset, current_task_idx, task_indices):
    """Evaluate test RMSE and NLPD for all specified tasks."""
    metrics = {}
    
    w=model.w if hasattr(model, 'w') else None

    # For each task we want to evaluate
    for task_idx in task_indices:
        # Load the dataset for this task
        if task_idx == current_task_idx:
            dataset = current_dataset
        else:
            dataset = ERA5Dataset(f"data/era5/processed_timeseries_4/task_{task_idx}")
        
        scale = dataset.scale
        scale = torch.tensor(scale).to(device)
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        task_rmse_list = []
        task_nlpd_list = []
        pi = torch.tensor(np.pi, device=device)

        print("Compute RMSE and NLPD")
        for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in enumerate(dataloader):
            # build tensors
            x_train = time_train.unsqueeze(-1).to(device)   # [B, T_train, 1]
            y_train = data_train.to(device)                 # [B, T_train, D]
            x_test  = time_test.unsqueeze(-1).to(device)    # [B, T_test, 1]
            y_test  = data_test.to(device)                  # [B, T_test, D]
            mask    = torch.ones_like(x_train)              # [B, T_train, 1]

            with torch.no_grad():
                if task_idx == 1:
                    if i == 0:
                        Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_test, w=w, recompute_kernel=True)
                    else:
                        Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_test, w=w, recompute_kernel=False)
                else:
                    if hasattr(model, 'Z_cache'):
                        Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_test, w=w, recompute_kernel=False)
                    else:
                        Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_test, w=w, recompute_kernel=True)
                mean_y, var_y, _, _ = model.pred_y_vmap(
                    y_train, mask, 100, Kxxs, Kxms, Kmms, Knms
                )  # mean_y, var_y: [B, T_test, D]

            # ---- RMSE per sequence ----
            err = (mean_y - y_test)  # [B, T_val, D]
            mse_seq   = err.pow(2).mean(dim=(1))           # [B, D]  
            rmse_seq  = torch.sqrt(mse_seq).mean(dim=(1))                  # [B]
            task_rmse_list.extend(rmse_seq.cpu().tolist())

            # ---- NLPD per sequence ----
            # elementwise negative log‐predictive‐density
            nlpd_elem = 0.5 * (
                torch.log(2*pi) + 
                torch.log(var_y) + 
                ( (y_test - mean_y)**2 ) / var_y
            )                                                # [B, T_test, D]
            nlpd_seq  = nlpd_elem.mean(dim=(1,2))            # [B]
            task_nlpd_list.extend(nlpd_seq.cpu().tolist())
        

        # finally, average across **all** sequences
        avg_rmse = float(np.mean(task_rmse_list))
        avg_nlpd = float(np.mean(task_nlpd_list))
        torch.cuda.empty_cache()

        print(f"Average RMSE per sequence: {avg_rmse:.4f}")
        print(f"Average NLPD per sequence: {avg_nlpd:.4f}")
        
        # Store metrics
        key = f"task_{current_task_idx}_eval_on_task_{task_idx}"
        metrics[key] = {
            "rmse": avg_rmse,
            "nlpd": avg_nlpd
        }
        
        print(f"Task {current_task_idx} evaluated on Task {task_idx}: RMSE = {avg_rmse:.4f}, NLPD = {avg_nlpd:.4f}")
        #torch.cuda.empty_cache()
    
    return metrics


def save_metrics_to_json(metrics):
    """Save metrics to a JSON file."""
    os.makedirs(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/metrics', exist_ok=True)
    with open(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to ./hsgpvae_{inducing_size}_{inducing_size}_{step_size}/metrics/test_metrics.json")


def plot_prediction_intervals(model, dataset, task_idx, dims=[0, 5, -1], grid_size=500, num_samples=100):
    """Plot smooth prediction intervals over a dense time grid."""
    device = next(model.parameters()).device

    # Retrieve dataset-wide scale & mean
    scale = torch.tensor(dataset.scale, device=device)
    mean_stat = torch.tensor(dataset.mean, device=device)

    # Prepare output folder
    out_dir = f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/plots/task_{task_idx}'
    os.makedirs(out_dir, exist_ok=True)

    # Pull one batch for this task
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_train, data_val, data_test, time_train, time_val, time_test, idx = next(iter(loader))

    # Build tensors
    x_train = time_train.unsqueeze(-1).to(device)   # shape: [1, T_train, 1]
    y_train = data_train.to(device)                 # shape: [1, T_train, D]
    x_test  = time_test.unsqueeze(-1).to(device)    # shape: [1, T_test, 1]
    y_test  = data_test.to(device)                  # shape: [1, T_test, D]

    # Define a dense, sorted time grid spanning train+test
    t_min = float(torch.cat([x_train, x_test], dim=1).min())
    t_max = float(torch.cat([x_train, x_test], dim=1).max())
    x_grid = torch.linspace(t_min, t_max, grid_size, device=device).view(1, grid_size, 1)
    mask_train = torch.ones_like(x_train)

    # Predict on the grid
    with torch.no_grad():
        w=model.w if hasattr(model, 'w') else None
        if task_idx == 1:
            Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_grid, w=w, recompute_kernel=True)
        else:
            if hasattr(model, 'Z_cache'):
                Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_grid, w=w, recompute_kernel=False)
            else:
                Kxxs, Kxms, Kmms, Knms = model.compute_kernels_val(x_train, x_grid, w=w, recompute_kernel=True)
        mean_grid, var_grid, _, _ = model.pred_y_vmap(
            y_train, mask_train, num_samples, 
            Kxxs, Kxms, Kmms, Knms
        )

    # Move everything to CPU & numpy
    x_train_np    = x_train.cpu().numpy()[0,:,0]
    y_train_np    = y_train.cpu().numpy()[0]           # shape [T_train, D]
    x_test_np     = x_test.cpu().numpy()[0,:,0]
    y_test_np     = y_test.cpu().numpy()[0]
    x_grid_np     = x_grid.cpu().numpy()[0,:,0]
    mean_grid_np  = mean_grid.cpu().numpy()[0]         # shape [grid_size, D]
    std_grid_np   = np.sqrt(var_grid.cpu().numpy()[0]) # shape [grid_size, D]

    # Rescale back to original units
    scale_np = scale.cpu().numpy()
    mean0_np = mean_stat.cpu().numpy()
    y_train_np   = y_train_np   * scale_np + mean0_np
    y_test_np    = y_test_np    * scale_np + mean0_np
    mean_grid_np = mean_grid_np * scale_np + mean0_np
    std_grid_np  = std_grid_np  * scale_np

    # Plot each requested dimension
    for dim in dims:
        if dim == -1:
            use_dim = mean_grid_np.shape[1] - 1
        else:
            use_dim = dim

        plt.figure(figsize=(10,6))
        # raw data
        plt.scatter(x_train_np, y_train_np[:, use_dim],
                    color='C0', alpha=0.4, label='Training', s=3)
        plt.scatter(x_test_np,  y_test_np[:,  use_dim],
                    color='C1', label='Test', s=3)

        # smooth mean & ribbon
        plt.plot(x_grid_np, mean_grid_np[:, use_dim],
                 'k-', label='Predicted Mean')
        plt.fill_between(
            x_grid_np,
            mean_grid_np[:, use_dim] - 1.96 * std_grid_np[:, use_dim],
            mean_grid_np[:, use_dim] + 1.96 * std_grid_np[:, use_dim],
            color='gray', alpha=0.3, label='95% CI'
        )

        plt.title(f'Task {task_idx} — Dimension {use_dim}')
        plt.xlabel('Time')
        plt.ylabel(f'Value (dim {use_dim})')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, f'dimension_{use_dim}.png'))
        plt.close()

    print(f"Plots saved for Task {task_idx} in {out_dir}")


def generate_summary_plots(metrics, num_tasks):
    """Generate summary plots for test metrics across all tasks."""
    # Create task grid for plotting
    task_grid = np.zeros((num_tasks, num_tasks))
    nlpd_grid = np.zeros((num_tasks, num_tasks))
    
    # Fill in the grid with metrics
    for curr_task in range(1, num_tasks+1):
        for eval_task in range(1, curr_task+1):
            key = f"task_{curr_task}_eval_on_task_{eval_task}"
            if key in metrics:
                task_grid[curr_task-1, eval_task-1] = metrics[key]["rmse"]
                nlpd_grid[curr_task-1, eval_task-1] = metrics[key]["nlpd"]
    
    # Create directory for summary plots
    os.makedirs(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/plots/summary', exist_ok=True)
    
    # Plot RMSE grid
    plt.figure(figsize=(10, 8))
    plt.imshow(task_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='RMSE')
    plt.title('Test RMSE across Tasks')
    plt.xlabel('Evaluated Task')
    plt.ylabel('Current Task')
    plt.xticks(np.arange(num_tasks), np.arange(1, num_tasks+1))
    plt.yticks(np.arange(num_tasks), np.arange(1, num_tasks+1))
    
    # Add values in the cells
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j <= i:  # Only show values for lower triangular part
                plt.text(j, i, f'{task_grid[i, j]:.4f}', 
                        ha='center', va='center', 
                        color='white' if task_grid[i, j] > np.mean(task_grid[task_grid > 0]) else 'black')
    
    plt.tight_layout()
    plt.savefig(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/plots/summary/rmse_grid.png')
    plt.close()
    
    # Plot NLPD grid
    plt.figure(figsize=(10, 8))
    plt.imshow(nlpd_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='NLPD')
    plt.title('Test NLPD across Tasks')
    plt.xlabel('Evaluated Task')
    plt.ylabel('Current Task')
    plt.xticks(np.arange(num_tasks), np.arange(1, num_tasks+1))
    plt.yticks(np.arange(num_tasks), np.arange(1, num_tasks+1))
    
    # Add values in the cells
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j <= i:  # Only show values for lower triangular part
                plt.text(j, i, f'{nlpd_grid[i, j]:.4f}', 
                        ha='center', va='center', 
                        color='white' if nlpd_grid[i, j] > np.mean(nlpd_grid[nlpd_grid > 0]) else 'black')
    
    plt.tight_layout()
    plt.savefig(f'./hsgpvae_{inducing_size}_{step_size}_seed{seed}/plots/summary/nlpd_grid.png')
    plt.close()
    
    print("Summary plots generated.")

def main():
    """Main function for the sequential learning tasks."""

    hippo = HiPPO_LegS(inducing_size, device)
    prev_discrete_steps = 0

    print("\n=== Running Task 1 ===")
    model1, task_data, end, num_discrete_steps, w = create_and_train_task1(hippo)
    prev_discrete_steps += num_discrete_steps

    # Calculate test metrics for Task 1
    test_metrics = evaluate_test_metrics(model1, task_data, 1, [1])
    
    # Save metrics to JSON
    save_metrics_to_json(test_metrics)
    
    # Plot predictions for selected dimensions
    plot_prediction_intervals(model1, task_data, 1, dims=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    # Store task 1 data for future reference
    model = model1
    all_test_metrics = test_metrics

    # Run remaining tasks
    num_tasks = 2  # Total number of tasks
    for task_idx in range(2, num_tasks+1):  # Tasks 2, 3, 4, 5
        start = end
        print(f"\n=== Running Task {task_idx} ===")
        model, task_data, num_discrete_steps, end = train_next_task(model, task_data, start,\
            task_idx, prev_discrete_steps, w, hippo)
        prev_discrete_steps += num_discrete_steps
        
        # Calculate test metrics for all previously seen tasks
        task_metrics = evaluate_test_metrics(model, task_data, task_idx, list(range(1, task_idx+1)))
        
        # Combine with all test metrics
        all_test_metrics.update(task_metrics)
        
        # Save metrics to JSON
        save_metrics_to_json(all_test_metrics)
        
        # Plot predictions for selected dimensions
        plot_prediction_intervals(model, task_data, task_idx, dims=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    
    # Generate summary plots
    generate_summary_plots(all_test_metrics, num_tasks)
    
    print("All tasks completed successfully!")


if __name__ == "__main__":
    seed_everything(seed)
    main()
