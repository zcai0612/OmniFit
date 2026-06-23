import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import Mlp
from src.models.layers.block import SelfAttnBlock
from src.models.heads.head_act import base_scale_act


class ScaleHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 516,
        hidden_dim: int = 256,
        num_layers: int = 3,
        act_type: str = "relu",
    ):
        super().__init__()
        
        # Feature aggregation: max + mean pooling
        self.pool_type = "max+mean"
        
        # MLP to predict scale
        mlp_dim_in = dim_in * 2 if self.pool_type == "max+mean" else dim_in
        
        layers = []
        in_dim = mlp_dim_in
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if act_type == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif act_type == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        
        # Final layer to predict scale (1 value)
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights for stable training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, point_feats):
        # point_feats: [B, N, C]
        B, N, C = point_feats.shape
        
        # Aggregate point features
        if self.pool_type == "max+mean":
            max_feat = torch.max(point_feats, dim=1)[0]  # [B, C]
            mean_feat = torch.mean(point_feats, dim=1)   # [B, C]
            global_feat = torch.cat([max_feat, mean_feat], dim=-1)  # [B, 2C]
        elif self.pool_type == "max":
            global_feat = torch.max(point_feats, dim=1)[0]  # [B, C]
        elif self.pool_type == "mean":
            global_feat = torch.mean(point_feats, dim=1)   # [B, C]
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        
        # Predict scale
        scale = self.mlp(global_feat)  # [B, 1]
        scale = scale.unsqueeze(-1)    # [B, 1, 1]
        
        # Apply activation to ensure positive scale
        scale = base_scale_act(scale)  # Using the imported activation function
        
        return scale
 
        