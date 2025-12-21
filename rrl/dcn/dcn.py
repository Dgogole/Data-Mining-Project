import torch
from torch import nn

class CrossLayer(nn.Module):
    
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.rand((n,)))
        self.b = nn.Parameter(torch.rand((n,)))
    
    def forward(self, x0, x):
        return x0 * (x @ self.w.unsqueeze(1)) + self.b.unsqueeze(0) + x

class DCN_net(nn.Module):

    def __init__(self, embedding_dim, deep_dim, layers_c, layers_d):
        super().__init__()
        self.cross = nn.ModuleList([CrossLayer(embedding_dim) for _ in range(layers_c)])
        self.deep = nn.Sequential(
            nn.Linear(embedding_dim, deep_dim),
            nn.BatchNorm1d(deep_dim),
            nn.ReLU(),
            *[layer for _ in range(layers_d-1) for layer in [nn.Linear(deep_dim, deep_dim), nn.BatchNorm1d(deep_dim), nn.ReLU()]])

    def forward(self, x):
        output_c = x
        for cross in self.cross:
            output_c = cross(x, output_c)
        output_h = self.deep(x)
        return torch.cat([output_c, output_h], dim=1)
    
class DCN_Encoder(nn.Module):
    def __init__(self, discrete_catenum):
        super().__init__()
        self.w = nn.ModuleList()
        self.discrete_dim = 0
        for c in discrete_catenum:
            dim = int(c ** 0.25 * 6)
            self.w.append(nn.Linear(c, dim, bias=False))
            self.discrete_dim += dim

    def forward(self, x):
        offset = 0
        features = []
        for layer in self.w:
            features.append(layer(x[:, offset:offset+layer.in_features]))
            offset += layer.in_features
        features.append(torch.log(x[:, offset:] + 1))
        return torch.cat(features, axis=1)

class DCN(nn.Module):
    def __init__(self, discrete_catenum, continuous_num, deep_dim, layers_c, layers_d, task_type):
        super().__init__()
        self.encoder = DCN_Encoder(discrete_catenum)
        embedding_dim = self.encoder.discrete_dim + continuous_num
        self.net = DCN_net(embedding_dim, deep_dim, layers_c, layers_d)
        feature_dim = embedding_dim + deep_dim
        if task_type == "regression":
            self.head = nn.Sequential(nn.Linear(feature_dim, 1, bias=False), nn.Softplus())
        elif task_type == "classification":
            self.head = nn.Sequential(nn.Linear(feature_dim, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        return self.head(self.net(x))
