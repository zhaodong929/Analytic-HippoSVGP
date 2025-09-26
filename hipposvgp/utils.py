import torch
import torch.nn as nn
import torch.nn.functional as F

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


if __name__ == "__main__":
    from kernel import SE_kernel
    mykernel = SE_kernel(input_dim=3)
    train_inputs = torch.randn(100,3)
    kernel_matrix = mykernel(train_inputs, train_inputs)
    ind_points =  pivoted_cholesky_init(train_inputs, kernel_matrix, 50)
    print(ind_points)
    print(ind_points.shape)
