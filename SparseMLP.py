import torch
import torch.nn as nn
import torch.nn.functional as F

from SparseLinear import SparseLinear


##Voir pour le learning rate et récompense
class SparseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sparsity=0.8, last_activation="elu"):
        super().__init__()
        # 4 premières couches avec ReLU
        self.fc1 = SparseLinear(input_dim, hidden_dim, sparsity)
        self.fc2 = SparseLinear(hidden_dim, hidden_dim, sparsity)
        self.fc3 = SparseLinear(hidden_dim, hidden_dim, sparsity)
        self.fc4 = SparseLinear(hidden_dim, hidden_dim, sparsity)
        self.fc5 = SparseLinear(hidden_dim, hidden_dim, sparsity)
        self.fc_out = SparseLinear(hidden_dim, output_dim, sparsity)

        # choisir l'activation de la dernière couche cachée
        if last_activation == "elu":
            self.last_activation = F.elu
        elif last_activation == "mish":
            self.last_activation = lambda x: x * torch.tanh(F.softplus(x))
        elif last_activation == "relu":
            self.last_activation = F.relu
        else:
            raise ValueError("last_activation doit être 'relu', 'elu' ou 'mish'")

    def forward(self, x):
        # 4 premières couches avec ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # dernière couche cachée avec activation personnalisée
        x = self.last_activation(self.fc5(x))
        # sortie (classification binaire, sigmoid à appliquer si besoin)
        x = self.fc_out(x)
        return x