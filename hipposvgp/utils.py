import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class PositiveDefiniteParameter(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Initialize an unconstrained lower triangular matrix
        self.unconstrained_L = nn.Parameter(torch.randn(size, size).double())
        # Mask to extract the lower triangular part
        self.register_buffer('tril_mask', torch.tril(torch.ones(size, size)) == 1)

    def forward(self):
        # Apply the mask to get a lower triangular matrix
        L = torch.zeros_like(self.unconstrained_L)
        L[self.tril_mask] = self.unconstrained_L[self.tril_mask]
        # Apply softplus to the diagonal to ensure positivity
        L = L - torch.diag(torch.diag(L)) + torch.diag(F.softplus(torch.diag(L)))
        return L

def pivoted_cholesky_init(
    train_inputs,
    kernel_matrix,
    max_length,
    epsilon=1e-10
    ):
    if train_inputs.shape[0] <= max_length:
        ind_points = train_inputs
    else:
        item_size = kernel_matrix.shape[-2]
        cis = torch.zeros((max_length, item_size)).to(kernel_matrix.device)
        di2s = kernel_matrix.diag()
        selected_items = []
        selected_item = torch.argmax(di2s)
        selected_items.append(selected_item)
        while len(selected_items) < max_length:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = torch.sqrt(di2s[selected_item])
            elements = kernel_matrix[..., selected_item, :]
            eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s = di2s - eis.pow(2.0)
            di2s[selected_item] = -(torch.tensor(float("inf")))
            selected_item = torch.argmax(di2s)
            #if di2s[selected_item] < epsilon:
            #    break
            selected_items.append(selected_item)
        ind_points = train_inputs[torch.stack(selected_items)]
    return ind_points


def pred_boundary(xx, yy, model, w=None):
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).double().to(model.device)
    with torch.no_grad():
        if w == None:
            mean, _ = model.pred_f(grid)
        else: 
            mean, _ = model.pred_f(grid, w)
        predicted_classes = (torch.sigmoid(mean) > 0.5)+0.
    return predicted_classes   

def sort_with_dimension(x, y, dim=0):
    ind = torch.argsort(x[:, dim])
    x = x[ind]
    y = y[ind]
    return x, y

def sort_with_l2(x, y):
    device = x.device
    N = len(x)
    res = []
    res_y = []
    ind = torch.argmin(torch.sum(x**2, 1)).item()
    cur = x[ind]
    cur_y = y[ind]
    res.append(cur.unsqueeze(0))
    res_y.append(cur_y.unsqueeze(0))
    if ind == 0:
        x = x[1:]
        y = y[1:]
    elif ind == len(x)-1:
        x = x[:-1]
        y = y[:-1]
    else:
        x = torch.concatenate([x[:ind], x[ind+1:]])
        y = torch.concatenate([y[:ind], y[ind+1:]])
    for i in range(N-1):
        ind = torch.argmin(torch.sum((x-cur.view(1, -1))**2, 1))
        cur = x[ind]
        cur_y = y[ind]
        res.append(cur.unsqueeze(0))
        res_y.append(cur_y.unsqueeze(0))
        cur_N = len(x)
        if ind == 0:
            x = x[1:]
            y = y[1:]
        elif ind == cur_N-1:
            x = x[:-1]
            y = y[:-1]
        else:
            x = torch.concatenate([x[:ind], x[ind+1:]])
            y = torch.concatenate([y[:ind], y[ind+1:]])
    return torch.concatenate(res).to(device), torch.concatenate(res_y).to(device)

def sort_with_l2_2(x, y):
    device = x.device
    N = len(x)
    res = []
    res_y = []
    ind = torch.argmax(torch.sum(x**2, 1)).item()
    cur = x[ind]
    cur_y = y[ind]
    res.append(cur.unsqueeze(0))
    res_y.append(cur_y.unsqueeze(0))
    if ind == 0:
        x = x[1:]
        y = y[1:]
    elif ind == len(x)-1:
        x = x[:-1]
        y = y[:-1]
    else:
        x = torch.concatenate([x[:ind], x[ind+1:]])
        y = torch.concatenate([y[:ind], y[ind+1:]])
    for i in range(N-1):
        ind = torch.argmax(torch.sum((x-cur.view(1, -1))**2, 1))
        cur = x[ind]
        cur_y = y[ind]
        res.append(cur.unsqueeze(0))
        res_y.append(cur_y.unsqueeze(0))
        cur_N = len(x)
        if ind == 0:
            x = x[1:]
            y = y[1:]
        elif ind == cur_N-1:
            x = x[:-1]
            y = y[:-1]
        else:
            x = torch.concatenate([x[:ind], x[ind+1:]])
            y = torch.concatenate([y[:ind], y[ind+1:]])
    return torch.concatenate(res).to(device), torch.concatenate(res_y).to(device)

def sort_with_se_kernel(x, y, pivot, log_ls, subsampling_lag=1, flag_min=True):
    ls = torch.exp(-log_ls)
    device = x.device
    N = len(x)
    res = []
    res_y = []
    x = x* ls.view(1, x.shape[1])
    if flag_min:
        ind = torch.argmin(torch.sum((x-pivot)**2, 1)).item()
    else:
        ind = torch.argmax(torch.sum((x-pivot)**2, 1)).item()
    cur = x[ind]
    cur_y = y[ind]
    res.append(cur.unsqueeze(0))
    res_y.append(cur_y.unsqueeze(0))
    if ind == 0:
        x = x[1:]
        y = y[1:]
    elif ind == len(x)-1:
        x = x[:-1]
        y = y[:-1]
    else:
        x = torch.concatenate([x[:ind], x[ind+1:]])
        y = torch.concatenate([y[:ind], y[ind+1:]])
    for i in range(N-1):
        ind = torch.argmin(torch.sum((x-cur.view(1, -1))**2, 1))
        cur = x[ind]
        cur_y = y[ind]
        res.append(cur.unsqueeze(0))
        res_y.append(cur_y.unsqueeze(0))
        cur_N = len(x)
        if ind == 0:
            x = x[1:]
            y = y[1:]
        elif ind == cur_N-1:
            x = x[:-1]
            y = y[:-1]
        else:
            x = torch.concatenate([x[:ind], x[ind+1:]])
            y = torch.concatenate([y[:ind], y[ind+1:]])
    return torch.concatenate(res)[::subsampling_lag].to(device), torch.concatenate(res_y)[::subsampling_lag].to(device)

def sort_with_se_kernel2(x, y, pivot, log_ls, subsampling_lag=1):
    ls = torch.exp(-log_ls)
    device = x.device
    N = len(x)
    res = []
    res_y = []
    x = x* ls.view(1, x.shape[1])

    ind = torch.argmax(torch.sum((x-pivot)**2, 1)).item()
    
    cur = x[ind]
    cur_y = y[ind]
    res.append(cur.unsqueeze(0))
    res_y.append(cur_y.unsqueeze(0))
    if ind == 0:
        x = x[1:]
        y = y[1:]
    elif ind == len(x)-1:
        x = x[:-1]
        y = y[:-1]
    else:
        x = torch.concatenate([x[:ind], x[ind+1:]])
        y = torch.concatenate([y[:ind], y[ind+1:]])
    for i in range(N-1):
        ind = torch.argmax(torch.sum((x-cur.view(1, -1))**2, 1))
        cur = x[ind]
        cur_y = y[ind]
        res.append(cur.unsqueeze(0))
        res_y.append(cur_y.unsqueeze(0))
        cur_N = len(x)
        if ind == 0:
            x = x[1:]
            y = y[1:]
        elif ind == cur_N-1:
            x = x[:-1]
            y = y[:-1]
        else:
            x = torch.concatenate([x[:ind], x[ind+1:]])
            y = torch.concatenate([y[:ind], y[ind+1:]])
    return torch.concatenate(res)[::subsampling_lag].to(device), torch.concatenate(res_y)[::subsampling_lag].to(device)



if __name__ == "__main__":
    from kernel import SE_kernel
    mykernel = SE_kernel(input_dim=3)
    train_inputs = torch.randn(100,3)
    kernel_matrix = mykernel(train_inputs, train_inputs)
    ind_points =  pivoted_cholesky_init(train_inputs, kernel_matrix, 50)
    print(ind_points)
    print(ind_points.shape)
