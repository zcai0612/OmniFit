import torch
import torch.nn as nn

from src.models.encoders.pointbert.custom_point_encoder import ScalePointTransformer
from src.models.layers.mlp import SimpleMLP



class ScalePredictor(nn.Module):
    def __init__(
        self,
        # Point Transformer
        trans_dim=516, 
        depth=12, 
        drop_path_rate=0.1, 
        num_heads=12, 
        group_size=32, 
        num_group=768, 
        encoder_dims=512, 
        # scale head
        head_hidden_dim=256,
        head_num_layers=2,
    ):
        super().__init__()

        self.point_encoder = ScalePointTransformer(
            trans_dim=trans_dim,
            depth=depth,
            drop_path_rate=drop_path_rate,
            num_heads=num_heads,
            group_size=group_size,
            num_group=num_group,
            encoder_dims=encoder_dims,
        )

        self.scale_head = SimpleMLP(
            input_dim=trans_dim,
            output_dim=1,
            hidden_dim=head_hidden_dim,
            num_layers=head_num_layers,
        )

        self.head_act = nn.ReLU(inplace=True)

    
    def forward(
        self,
        pcds, # [B, N, 4]
    ):
        scale_tokens = self.point_encoder(pcds)  # [B, 1, C]
        scale = self.scale_head(scale_tokens)  # [B, 1, 1]
        scale = self.head_act(scale)
        return scale