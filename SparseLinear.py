import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.8):
        super().__init__()

        # Initialisation des poids
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Initialisation des biais
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Masque binaire permettant de supprimer les liens neuronnaux (res = 0 || 1)
        self.mask = nn.Parameter(
            (torch.rand(out_features, in_features) > sparsity).float(),
            requires_grad=False
        )

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)