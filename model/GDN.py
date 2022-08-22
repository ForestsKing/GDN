import math

import torch
from torch import nn
import torch.nn.functional as F

from model.GNN import GNN


class GDN(nn.Module):
    def __init__(self, feature_num, timd_num=5, embed_dim=64, topk=20):
        super(GDN, self).__init__()
        self.topk = topk + 1  # 加上自身
        self.embedding = nn.Embedding(feature_num, embed_dim)
        self.graph = GNN(timd_num, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

        self.dp = nn.Dropout(0.2)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x):
        batch_num, feature_num, time_num = x.shape
        device = x.device

        # Sensor Embedding
        embedding = self.embedding(torch.arange(feature_num).to(device))

        # Graph Structure Learning
        cos_ji_mat = torch.matmul(embedding, embedding.T)
        normed_mat = torch.matmul(embedding.norm(dim=-1).view(-1, 1), embedding.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat

        topk_ji_idx = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]  # 绝对值大小？
        topk_ji = torch.zeros_like(cos_ji_mat).to(x.device)
        for i in range(len(topk_ji)):
            topk_ji[i][topk_ji_idx[i]] = 1

        # Graph Attention-Based Forecasting
        ## Feature Extractor
        z = self.graph(x, embedding, topk_ji)

        ## Output Layer
        out = z * embedding

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)

        out = self.fc(out).view(-1, feature_num)
        return out
