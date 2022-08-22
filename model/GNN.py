import math

import torch
from torch import nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, timd_num=5, embed_dim=64, alpha=0.2):
        super(GNN, self).__init__()
        self.alpha = alpha
        self.W = nn.Linear(timd_num, embed_dim)
        self.A = nn.Linear(4 * embed_dim, 1)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.W.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.A.weight, gain=math.sqrt(2))

    def forward(self, x, embedding, mask):
        x = self.W(x)
        g = torch.cat((x, embedding.unsqueeze(0).repeat(x.shape[0], 1, 1)), dim=-1)

        gi = g.unsqueeze(2).repeat(1, 1, g.shape[1], 1)
        gj = g.unsqueeze(1).repeat(1, g.shape[1], 1, 1)
        gij = torch.cat((gi, gj), dim=-1)
        e = self.A(gij).squeeze()
        e = F.leaky_relu(e, negative_slope=self.alpha)

        zero_vec = -9e15 * torch.ones_like(e).to(x.device)
        a = torch.where(mask > 0, e, zero_vec)
        a = F.softmax(a, dim=-1)

        z = torch.matmul(a, x)
        z = F.relu(z)
        return z
