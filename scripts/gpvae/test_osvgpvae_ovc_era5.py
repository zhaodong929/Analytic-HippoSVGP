#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HiPPO-SVGP Sequential Learning for Solar Data
This script implements sequential learning tasks using HiPPO-SVGP models.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from hipposvgp.svgpvae import SVGPVAEMultipleKernels, SVGPVAEMultipleKernels_EWC
from scripts.gpvae.data import ERA5Dataset
from hipposvgp.utils import pivoted_cholesky_init
import random

torch.autograd.set_detect_anomaly(True) 

import argparse
parser = argparse.ArgumentParser(description='HiPPO-SVGP for Solar Data')
parser.add_argument('--inducing_size', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
seed = args.seed

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
inducing_size = args.inducing_size
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


def create_and_train_task1():
    """Create and train the model for Task 1."""
    dataset = ERA5Dataset("data/era5/processed_timeseries_4/task_1")
    # start = dataset[0][]
    # start = 0
    # end = 930
    start = int(min(dataset[0][-4]))
    end = int(max(dataset[0][-4]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create kernels
    kernel = [
        SE_kernel(1, device=device)
        for _ in range(dim_f)
    ]

    # Create encoder and decoder networks
    encoder = MLPEncoder(dim_y=dim_y, dim_f=dim_f, num_hidden_units=16).to(device)
    decoder = MLPDecoder(dim_f=dim_f, dim_y=dim_y, num_hidden_units=16).to(device)
    
    # Create inducing points
    inducing_points = torch.linspace(start, end, inducing_size)[:, None].to(device)

    # Create model for task 1
    model = SVGPVAEMultipleKernels(
        kernel=kernel,
        likelihood=GaussianLikelihood(0.01),
        encoder=encoder,
        decoder=decoder,
        inducing_points=inducing_points,
        num_input_axes=num_input_axes,
        N_train=dataset[0][0].shape[-2],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-3)
    num_epochs = 20
    
    best_checkpoint = None  # To store the best checkpoint
    best_val_rmse = float('inf')
    scale = dataset.scale
    scale = torch.tensor(scale).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        for data_train, data_val, data_test, time_train, time_val, time_test, idx in tqdm(dataloader):
            x_batch = time_train.unsqueeze(2).to(device)  # [bs, N, 1]
            y_batch = data_train.to(device)  # [bs, N, ydim]
            mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
            seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
            optimizer.zero_grad()
            loss = -model.elbo(
                x_batch,
                y_batch,
                mask_batch,
                seq_length_batch,
                torch.zeros((inducing_size, 1, dim_f, x_batch.shape[0]), device=device),
                torch.zeros((inducing_size, inducing_size, dim_f, x_batch.shape[0]), device=device)
            )
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item() * x_batch.shape[0])
        loss_mean = np.sum(np.array(loss_list)) / len(dataset)
        
        # Calculate validation RMSE
        with torch.no_grad():
            model.eval()
            val_rmse_list = []
            for data_train, data_val, data_test, time_train, time_val, time_test, idx in dataloader:
                x_train_batch = time_train.unsqueeze(2).to(device)
                y_train_batch = data_train.to(device)
                x_val_batch = time_val.unsqueeze(2).to(device)
                y_val_batch = data_val.to(device)
                mask_batch = torch.ones(x_train_batch.shape[0], x_train_batch.shape[1], 1).to(device).float()
                seq_length_batch = torch.tensor([x_train_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_train_batch.shape[0], 1))
                pred_y, _, _, _ = model.pred_y_vmap(x_train_batch, y_train_batch, x_val_batch, mask_batch, num_samples=100)
                assert pred_y.shape == y_val_batch.shape
                
                # Calculate RMSE with dataset.scale (x - mean) / scale
                # so (x*scale+mean) - (y*scale+mean) = (x-y)*scale
                # undo normalization and compute per-sequence MSE
                err = (pred_y - y_val_batch)               # still [B, T_val, D]
                mse_per_seq = err.pow(2).mean(dim=(1))        # [B, D]
                rmse_per_seq = torch.sqrt(mse_per_seq).mean(dim=(1))  # [B]

                # collect each sequence’s RMSE
                val_rmse_list.extend(rmse_per_seq.cpu().tolist())

            # now average across all sequences
            val_rmse_mean = float(np.mean(val_rmse_list))
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_mean}, Validation RMSE: {val_rmse_mean}")
            torch.cuda.empty_cache()
        
        # Save the best checkpoint if validation improves
        if val_rmse_mean < best_val_rmse:
            best_val_rmse = val_rmse_mean
            best_checkpoint = f'./svgpvaeOVC_{inducing_size}_seed{seed}/task1/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)
            torch.save(model.state_dict(), best_checkpoint)

    # Load the best model and save final version
    model.load_state_dict(torch.load(best_checkpoint))
    best_model_path = f'./svgpvaeOVC_{inducing_size}_seed{seed}/task1/model.best.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    
    # Calculate and save test metrics
    test_metrics = evaluate_test_metrics(model, dataset, 1, [1])
    
    # Plot predictions for selected dimensions
    plot_prediction_intervals(model, dataset, 1, dims=[0, 5, -1])

    # Save test metrics to JSON
    save_metrics_to_json(test_metrics)
    
    return model, dataset, test_metrics


@torch.no_grad()
def batch_params(model, dataset):
    """Compute batch parameters for the model using torch.vmap."""
    def _batch_fn(x_train, y_train, mask):
        qnet_mu_old, qnet_var_old = model.encoder(y_train)
        Knms_old = torch.stack([model.kernel[i](x_train, model.inducing_points) 
                                 for i in range(model.dim_f)])
        Ku_old_old = torch.stack([model.kernel[i](model.inducing_points) 
                                  for i in range(model.dim_f)])
        mu_hat_old, A_hat_old = model.approximate_posterior_inducing_params(
            mask,
            qnet_mu_old,
            qnet_var_old,
            Ku_old_old,
            Knms_old,
        )
        return mu_hat_old, A_hat_old

    vmap_fn = torch.vmap(_batch_fn, in_dims=(0, 0, 0), out_dims=(-1, -1))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    res_mu_old = []
    res_A_old = []
    for _, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in enumerate(dataloader):
        x_batch = time_train.unsqueeze(2).to(device)  # [bs, N, 1]
        y_batch = data_train.to(device)  # [bs, N, ydim]
        mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
        mu_hat_old_batch, A_hat_old_batch = vmap_fn(x_batch, y_batch, mask_batch)
        res_mu_old.append(mu_hat_old_batch)
        res_A_old.append(A_hat_old_batch)
    return torch.concatenate(res_mu_old, dim=-1), torch.concatenate(res_A_old, dim=-1)


def train_next_task(prev_model, prev_data, task_idx, all_test_metrics):
    """Train model for task_idx (2-10) with EWC."""
    # Extract parameters for continual learning
    mu_hat_old, A_hat_old = batch_params(prev_model, prev_data)
    print(f"mu_hat_old shape: {mu_hat_old.shape}, A_hat_old shape: {A_hat_old.shape}")
    mu_hat_old_np = mu_hat_old.detach().cpu().numpy()
    A_hat_old_np = A_hat_old.detach().cpu().numpy()
    os.makedirs(f'./svgpvaeOVC_{inducing_size}_seed{seed}/task{task_idx}', exist_ok=True)
    np.savez(f'./svgpvaeOVC_{inducing_size}_seed{seed}/task{task_idx}/mu_A_old.npz', mu_hat_old=mu_hat_old_np, A_hat_old=A_hat_old_np)
    
    inducing_points_old = prev_model.inducing_points.clone()
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
    
    # Prepare data for current task
    dataset = ERA5Dataset(f"data/era5/processed_timeseries_4/task_{task_idx}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # Create inducing points - CRITICAL: use start from task1
    M_old = int((task_idx-1)/(task_idx)* inducing_size)
    M_new = inducing_size - M_old
    old_Z = inducing_points_old[torch.randperm(inducing_size)[0:M_old], :]
    start_task = np.min(np.array([dataset[i][4] for i in range(len(dataset))]))
    end_task = np.max(np.array([dataset[i][4] for i in range(len(dataset))]))
    new_Z = torch.linspace(start_task, end_task, M_new)[:, None].to(device)
    inducing_points_task = torch.vstack((old_Z, new_Z))
    


    # Compute Fisher Information and store optimal parameters
    optimal_params = {
        name: param.clone().detach()
        for name, param in prev_model.named_parameters()
        if "encoder" in name or "decoder" in name
    }

    prev_dataloader = DataLoader(prev_data, batch_size=5, shuffle=False)
    fisher_info = {
        name: 0
        for name, param in prev_model.named_parameters()
        if param.requires_grad and ("encoder" in name or "decoder" in name)
    }
    print("Computing Fisher Information...")
    for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in tqdm(enumerate(prev_dataloader)): 
        prev_model.zero_grad()
        x_batch = time_train.unsqueeze(2).to(device)
        y_batch = data_train.to(device)
        mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
        seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
        if task_idx == 1:
            mu_hat_old_batch = torch.zeros((inducing_size, 1, dim_f, x_batch.shape[0]), device=device)
            A_hat_old_batch = torch.zeros((inducing_size, inducing_size, dim_f, x_batch.shape[0]), device=device)
        else:
            mu_hat_old_batch = mu_hat_old[..., idx]
            A_hat_old_batch = A_hat_old[..., idx]

        if task_idx == 2:
            loss_prev_task = -prev_model.elbo(
                x_batch,
                y_batch,
                mask_batch,
                seq_length_batch,
                mu_hat_old_batch,
                A_hat_old_batch,
            )
        else:
            loss_prev_task = -prev_model.elbo(
                x_batch,
                y_batch,
                mask_batch,
                seq_length_batch,
                mu_hat_old_batch,
                A_hat_old_batch,
                add_ewc=False
            )

        for j in range(len(loss_prev_task)):
            loss_prev_task[j].backward(retain_graph=True)
            for name, param in prev_model.named_parameters():
                if param.requires_grad and ("encoder" in name or "decoder" in name):
                    fisher_info[name] += param.grad**2 / len(prev_data)
            prev_model.zero_grad()
        torch.cuda.empty_cache()
    

    # Create EWC model for current task
    model_task = SVGPVAEMultipleKernels_EWC(
        kernel=kernel_old,
        likelihood=likelihood_task,
        encoder=encoder_task,
        decoder=decoder_task,
        inducing_points=inducing_points_task,
        num_input_axes=num_input_axes,
        N_train=dataset[0][0].shape[-2],
        mu_hat_old=mu_hat_old, 
        A_hat_old=A_hat_old,    
        encoder_old=encoder_old,
        inducing_points_old=inducing_points_old,
        lambda_ewc = lambda_ewc
    )
    
    Z_old = prev_model.inducing_points
    M = Z_old.shape[0]
    old_Z = Z_old.clone()
    new_Z = torch.linspace(start_task, end_task, M_new)[:, None].to(device)
    Z_input = torch.vstack((old_Z, new_Z))
    kernel_matrices = torch.stack(
        [model_task.kernel[i](Z_input, Z_input) for i in range(model_task.dim_f)]
    )

    Z_list = []
    for i in range(model_task.dim_f):
        Z = pivoted_cholesky_init(Z_input, kernel_matrices[i], min(inducing_size, Z_input.shape[0]))
        Z_list.append(Z)
    Z = torch.vstack(Z_list)
    Z_numpy = Z.detach().cpu().numpy()  
    # use kmeans with 1 cluster to pick centre as Z with sklearn 
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=inducing_size, random_state=seed).fit(Z_numpy)
    inducing_points = torch.tensor(kmeans.cluster_centers_, device=device)
    model_task.inducing_points = torch.nn.Parameter(inducing_points)
    model_task.inducing_points.requires_grad = False

    # Load optimal parameters and Fisher Information
    model_task.optimal_params = optimal_params
    model_task.fisher_info = fisher_info
    
    optimizer = Adam(model_task.parameters(), lr=5e-3)
    num_epochs = 20
    best_checkpoint = None
    best_val_rmse = float('inf')

    scale = dataset.scale
    scale = torch.tensor(scale).to(device)
    
    print(f"Training Task {task_idx}")
    for epoch in range(num_epochs):
        model_task.train()
        loss_list = []
        for i, (data_train, data_val, data_test, time_train, time_val, time_test, idx) in tqdm(enumerate(dataloader)):
            x_batch = time_train.unsqueeze(2).to(device)
            y_batch = data_train.to(device)
            mask_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], 1).to(device).float()
            seq_length_batch = torch.tensor([x_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_batch.shape[0], 1))
            optimizer.zero_grad()
            
            mu_hat_old_batch = model_task.mu_hat_old[..., idx]
            A_hat_old_batch = model_task.A_hat_old[..., idx]
            loss = -model_task.elbo(x_batch, y_batch, mask_batch, seq_length_batch, mu_hat_old_batch, A_hat_old_batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item() * x_batch.shape[0])
        loss_mean = np.sum(np.array(loss_list)) / len(dataset)
        
        # Calculate validation RMSE
        with torch.no_grad():
            model_task.eval()
            val_rmse_list = []
            for data_train, data_val, data_test, time_train, time_val, time_test, idx in dataloader:
                x_train_batch = time_train.unsqueeze(2).to(device)
                y_train_batch = data_train.to(device)
                x_val_batch = time_val.unsqueeze(2).to(device)
                y_val_batch = data_val.to(device)
                mask_batch = torch.ones(x_train_batch.shape[0], x_train_batch.shape[1], 1).to(device).float()
                seq_length_batch = torch.tensor([x_train_batch.shape[1]]).to(device).float().unsqueeze(1).tile((x_train_batch.shape[0], 1))
                pred_y, _, _, _ = model_task.pred_y_vmap(x_train_batch, y_train_batch, x_val_batch, mask_batch, num_samples=100)
                assert pred_y.shape == y_val_batch.shape
                # Calculate RMSE with dataset.scale (x - mean) / scale
                # so (x*scale+mean) - (y*scale+mean) = (x-y)*scale
                # undo normalization and compute per-sequence MSE
                # Calculate RMSE with dataset.scale 
                err = (pred_y - y_val_batch)  # [B, T_val, D]
                mse_per_seq = err.pow(2).mean(dim=(1))  # [B, D]
                rmse_per_seq = torch.sqrt(mse_per_seq).mean(dim=(1))  # [B]

                # collect each sequence’s RMSE
                val_rmse_list.extend(rmse_per_seq.cpu().tolist())

            # now average across all sequences
            val_rmse_mean = float(np.mean(val_rmse_list))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_mean}, Validation RMSE: {val_rmse_mean}")
        
        # Save the best checkpoint if validation improves
        if val_rmse_mean < best_val_rmse:
            best_val_rmse = val_rmse_mean
            best_checkpoint = f'./svgpvaeOVC_{inducing_size}_seed{seed}/task{task_idx}/checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)
            torch.save(model_task.state_dict(), best_checkpoint)
        torch.cuda.empty_cache()
    
    # Load the best model and save final version
    model_task.load_state_dict(torch.load(best_checkpoint))
    best_model_path = f'./svgpvaeOVC_{inducing_size}_seed{seed}/task{task_idx}/model.best.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model_task.state_dict(), best_model_path)
    
    # Calculate test metrics for all previously seen tasks
    task_metrics = evaluate_test_metrics(model_task, dataset, task_idx, list(range(1, task_idx+1)))
    
    # Combine with all test metrics
    all_test_metrics.update(task_metrics)
    
    # Save metrics to JSON
    save_metrics_to_json(all_test_metrics)
    
    # Plot predictions for selected dimensions
    plot_prediction_intervals(model_task, dataset, task_idx, dims=[0, 5, -1])
    
    return model_task, dataset, all_test_metrics


@torch.no_grad()
def evaluate_test_metrics(model, current_dataset, current_task_idx, task_indices):
    """Evaluate test RMSE and NLPD for all specified tasks."""
    metrics = {}
    
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

        # assume `scale` is a 1‑D tensor of length D, already on `device`

        for data_train, data_val, data_test, time_train, time_val, time_test, idx in dataloader:
            # build tensors
            x_train = time_train.unsqueeze(-1).to(device)   # [B, T_train, 1]
            y_train = data_train.to(device)                 # [B, T_train, D]
            x_test  = time_test.unsqueeze(-1).to(device)    # [B, T_test, 1]
            y_test  = data_test.to(device)                  # [B, T_test, D]
            mask    = torch.ones_like(x_train)              # [B, T_train, 1]

            with torch.no_grad():
                mean_y, var_y, _, _ = model.pred_y_vmap(
                    x_train, y_train, x_test, mask, num_samples=100
                )  # mean_y, var_y: [B, T_test, D]

            # ---- RMSE per sequence ----
            err = (mean_y - y_test)                  # [B, T_test, D]
            mse_seq   = err.pow(2).mean(dim=(1))           # [B]  mean over time & dims
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

        print(f"Average RMSE per sequence: {avg_rmse:.4f}")
        print(f"Average NLPD per sequence: {avg_nlpd:.4f}")
        
        # Store metrics
        key = f"task_{current_task_idx}_eval_on_task_{task_idx}"
        metrics[key] = {
            "rmse": avg_rmse,
            "nlpd": avg_nlpd
        }
        
        print(f"Task {current_task_idx} evaluated on Task {task_idx}: RMSE = {avg_rmse:.4f}, NLPD = {avg_nlpd:.4f}")
    
    return metrics


def save_metrics_to_json(metrics):
    """Save metrics to a JSON file."""
    os.makedirs(f'./svgpvaeOVC_{inducing_size}_seed{seed}/metrics', exist_ok=True)
    with open(f'./svgpvaeOVC_{inducing_size}_seed{seed}/metrics/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to ./svgpvaeOVC_{inducing_size}_seed{seed}/metrics/test_metrics.json")


def plot_prediction_intervals(model, dataset, task_idx, dims=[0, 5, -1], grid_size=500, num_samples=100):
    """Plot smooth prediction intervals over a dense time grid."""
    device = next(model.parameters()).device

    # Retrieve dataset-wide scale & mean
    scale = torch.tensor(dataset.scale, device=device)
    mean_stat = torch.tensor(dataset.mean, device=device)

    # Prepare output folder
    out_dir = f'./svgpvaeOVC_{inducing_size}_seed{seed}/plots/task_{task_idx}'
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
        mean_grid, var_grid, _, _ = model.pred_y_vmap(
            x_train, y_train, x_grid, mask_train, num_samples=num_samples
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


def main():
    """Main function for the sequential learning tasks."""
    
    print("\n=== Running Task 1 ===")
    model1, task_data, test_metrics = create_and_train_task1()
    model = model1  # Store task 1 model for future tasks
    all_test_metrics = test_metrics
    
    # Run tasks 2-10 sequentially
    num_tasks = 2  # Total number of tasks
    for task_idx in range(2, num_tasks+1):
        print(f"\n=== Running Task {task_idx} ===")
        model, task_data, all_test_metrics = train_next_task(model, task_data, task_idx, all_test_metrics)
    
    # Generate summary metrics and plots
    generate_summary_plots(all_test_metrics, num_tasks)
    
    print("All tasks completed successfully!")


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
    os.makedirs(f'./svgpvaeOVC_{inducing_size}_seed{seed}/plots/summary', exist_ok=True)
    
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
    plt.savefig(f'./svgpvaeOVC_{inducing_size}_seed{seed}/plots/summary/rmse_grid.png')
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
    plt.savefig(f'./svgpvaeOVC_{inducing_size}_seed{seed}/plots/summary/nlpd_grid.png')
    plt.close()
    
    print("Summary plots generated.")


if __name__ == "__main__":
    # seed_everything()
    # 42, 0, 1
    seed_everything(seed)
    main()
