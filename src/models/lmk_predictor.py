import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from src.models.layers.block import SelfAttnBlock, CrossAttnBlock
from src.models.layers.mlp import SimpleMLP
from src.models.heads.scale_head import ScaleHead


class LmkPredictor(nn.Module):
    def __init__(
        self,
        num_lmks=600,
        embed_dim=512,
        point_feat_dim=516,
        pixel_feat_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        init_values=0.01,
        use_pixel_adapter=False,
        pixel_adapter_ratio=0.5,
        adapter_type="parallel",
    ):
        super().__init__()
        
        self.point_feat_dim = point_feat_dim
        self.pixel_feat_dim = pixel_feat_dim
        self.depth = depth
        self.use_pixel_adapter = use_pixel_adapter
        self.pixel_adapter_ratio = pixel_adapter_ratio
        
        if use_pixel_adapter:
            self.pixel_feat_proj = nn.Linear(pixel_feat_dim, embed_dim)
        else:
            self.pixel_feat_proj = None

        self.point_feat_proj = nn.Linear(point_feat_dim, embed_dim)
        self.cross_attn_blocks = nn.ModuleList([
            self.get_attn_block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=None,
                aa_type='cross',
                use_pixel_adapter=use_pixel_adapter,
                pixel_adapter_ratio=pixel_adapter_ratio,
                adapter_type=adapter_type
            )
            for _ in range(depth)
        ])

        self.self_attn_blocks = nn.ModuleList([
            self.get_attn_block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=None,
                aa_type='self',
            )
            for _ in range(depth)
        ])

        self.lmk_tokens = nn.Parameter(torch.randn(1, num_lmks, embed_dim))
        nn.init.normal_(self.lmk_tokens, std=1e-6)

        self.lmk_head = SimpleMLP(
            input_dim=embed_dim,
            output_dim=3,
            hidden_dim=embed_dim,
            num_layers=2,
        )

        self.use_reentrant = False

    def _process_attention(self, x, point_feats, pixel_feats, q_pos, k_pos):
        tokens = x
        for i in range(self.depth):
            if self.training:
                tokens = checkpoint(self.cross_attn_blocks[i], tokens, point_feats, pixel_feats, q_pos, k_pos, use_reentrant=self.use_reentrant)
                tokens = checkpoint(self.self_attn_blocks[i], tokens, q_pos, use_reentrant=self.use_reentrant)
            else: 
                tokens = self.cross_attn_blocks[i](tokens, point_feats, pixel_feats, q_pos, k_pos)
                tokens = self.self_attn_blocks[i](tokens, q_pos)
        return tokens

    def get_attn_block(
        self, 
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        proj_bias,
        ffn_bias,
        init_values,
        qk_norm,
        rope,
        aa_type,
        use_pixel_adapter=False,
        pixel_adapter_ratio=0.5,
        adapter_type='parallel',
    ):
        if aa_type == 'self':
            attn_block = SelfAttnBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                norm_layer=nn.LayerNorm,
                qk_norm=qk_norm,
                rope=rope,
            )
        elif aa_type == 'cross':
            attn_block = CrossAttnBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                norm_layer=nn.LayerNorm,
                qk_norm=qk_norm,
                rope=rope,
                use_pixel_adapter=use_pixel_adapter,
                pixel_adapter_ratio=pixel_adapter_ratio,
                adapter_type=adapter_type
                # fused_attn=False
            )
        return attn_block
    
    def forward(
        self,
        point_feats: torch.Tensor, # [B, N, C]
        pixel_feats: torch.Tensor=None, # [B, N, C]
    ):
        B = point_feats.shape[0]
        point_feats = self.point_feat_proj(point_feats)  # [B, N, C]
        if self.use_pixel_adapter and pixel_feats is not None:
            pixel_feats = self.pixel_feat_proj(pixel_feats)  # [B, N, C]
        else:
            pixel_feats = None
    
        lmk_tokens = self.lmk_tokens.repeat(B, 1, 1)  # [B, num_lmks, C]
        lmk_tokens = self._process_attention(lmk_tokens, point_feats, pixel_feats, None, None)  # [B, num_lmks, C]
        pred_lmks = self.lmk_head(lmk_tokens)  # [B, num_lmks, 3]

        return pred_lmks

    def load_weights(self, lmk_predictor_weight_path: str, pixel_adapter_weight_path: str=None):
        state_dict = self.state_dict()
        
        new_state_dict = {}
        lmk_predictor_state_dict = torch.load(lmk_predictor_weight_path, map_location='cpu', weights_only=True)
        new_state_dict.update(lmk_predictor_state_dict)
        
        if self.use_pixel_adapter:
            if pixel_adapter_weight_path is not None:
                pixel_adapter_state_dict = torch.load(pixel_adapter_weight_path, map_location='cpu', weights_only=True)
                new_state_dict.update(pixel_adapter_state_dict)
            else:
                for key in state_dict.keys():
                    if "pixel" in key:
                        if "out_proj_pixel" in key:
                            new_state_dict[key] = torch.zeros_like(state_dict[key])
                        else:
                            new_state_dict[key] = state_dict[key].clone()
        self.load_state_dict(new_state_dict)

    def save_weights(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_weights_path = os.path.join(output_dir, 'lmk_predictor.pt')
        pixel_adapter_weights_path = os.path.join(output_dir, 'pixel_adapter.pt')
        
        state_dict = self.state_dict()
        if not self.use_pixel_adapter:
            torch.save(state_dict, model_weights_path)
            return
        else:
            model_state_dict = {}
            pixel_adapter_state_dict = {}
            for key in state_dict.keys():
                if "pixel" in key:
                    pixel_adapter_state_dict[key] = state_dict[key].clone()
                else:
                    model_state_dict[key] = state_dict[key].clone()
            torch.save(model_state_dict, model_weights_path)
            torch.save(pixel_adapter_state_dict, pixel_adapter_weights_path)
            return
    
    def save_pixel_adapter_weights(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        pixel_adapter_weights_path = os.path.join(output_dir, 'pixel_adapter.pt')
        state_dict = self.state_dict()
        pixel_adapter_state_dict = {}
        for key in state_dict.keys():
            if "pixel" in key:
                pixel_adapter_state_dict[key] = state_dict[key].clone()
        torch.save(pixel_adapter_state_dict, pixel_adapter_weights_path)