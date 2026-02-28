"""Shared utilities: parameter constraints and inducing point initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveDefiniteParameter(nn.Module):
    """Parameterize a positive definite matrix via its Cholesky factor.

    The diagonal is passed through softplus to ensure positivity.
    """

    def __init__(self, size):
        super().__init__()
        self.unconstrained_L = nn.Parameter(torch.randn(size, size).double())
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(size, size)) == 1
        )

    def forward(self):
        L = torch.zeros_like(self.unconstrained_L)
        L[self.tril_mask] = self.unconstrained_L[self.tril_mask]
        L = L - torch.diag(torch.diag(L)) + torch.diag(
            F.softplus(torch.diag(L))
        )
        return L


def pivoted_cholesky_init(train_inputs, kernel_matrix, max_length, epsilon=1e-10):
    """Select inducing points via pivoted Cholesky decomposition.

    Args:
        train_inputs: [N, D] training data
        kernel_matrix: [N, N] kernel matrix of training data
        max_length: number of inducing points to select
        epsilon: numerical tolerance

    Returns:
        Selected inducing points [max_length, D].
    """
    if train_inputs.shape[0] <= max_length:
        return train_inputs

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
        selected_items.append(selected_item)

    return train_inputs[torch.stack(selected_items)]
