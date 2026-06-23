# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange

XFORMERS_AVAILABLE = False


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MemEffSelfAttention(SelfAttention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x    

class CrossAttention(nn.Module):
    """
    Cross Attention layer, query from x, key and value from context
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        use_pixel_adapter=False,
        pixel_adapter_ratio=0.5,
        adapter_type="parallel", # serial or parallel
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
        self.use_pixel_adapter = use_pixel_adapter
        self.pixel_adapter_ratio = pixel_adapter_ratio
        self.adapter_type = adapter_type
        
        if use_pixel_adapter:
            self.q_proj_pixel = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv_proj_pixel = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.out_proj_pixel = nn.Linear(dim, dim, bias=proj_bias)
            self.out_drop_pixel = nn.Dropout(proj_drop)
        else:
            self.q_proj_pixel = None
            self.kv_proj_pixel = None
            self.out_proj_pixel = None
            self.out_drop_pixel = None

    def forward(
        self, 
        x: Tensor, 
        point_feats: Tensor, pixel_feats: Tensor=None, 
        q_pos=None, k_pos=None
    ) -> Tensor:
        """
        Args:
            x: query [B, N_q, C]
            point_feats: key and value [B, N_k, C]
            pixel_feats: pixel adapter features [B, N_k, C] (optional)
            q_pos: query position encoding
            k_pos: key position encoding
        Returns:
            output [B, N_q, C]
        """
        B, N_q, C = x.shape
        B_p, N_p, C_p = point_feats.shape
        
        x_in = x.clone()
        
        assert B == B_p, "x and point feats must have the same batch size"
        assert C == C_p, "x and point feats must have the same feature dimension"
        
        # compute query
        q = self.q_proj(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # compute key and value
        kv = self.kv_proj(point_feats).reshape(B, N_p, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)

        # apply rotary position encoding
        if self.rope is not None:
            if q_pos is not None:
                q = self.rope(q, q_pos)
            if k_pos is not None:
                k = self.rope(k, k_pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, num_heads, N_q, N_k]
            
            # attn_save_path = None
            # for i in range(24):
            #     attn_save_path = os.path.join("outputs_vis/0000_00006_02_00021", f"attn_{i:02}.pt")
            #     if os.path.exists(attn_save_path):
            #         pass
            #     else:
            #         break
            # torch.save(attn.detach().cpu(), attn_save_path)
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, num_heads, N_q, head_dim]

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # process pixel adapter if enabled
        if self.use_pixel_adapter and pixel_feats is not None:
            B_i, N_i, C_i = pixel_feats.shape
            assert B == B_i, "x and pixel_feats must have the same batch size"
            assert C == C_i, "x and pixel_feats must have the same feature dimension"
                
            if self.adapter_type == "parallel":
                q_pixel = self.q_proj_pixel(x_in).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                kv_pixel = self.kv_proj_pixel(pixel_feats).reshape(B, N_i, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k_pixel, v_pixel = kv_pixel.unbind(0)
                x_pixel = F.scaled_dot_product_attention(q_pixel, k_pixel, v_pixel, dropout_p=self.attn_drop.p if self.training else 0.0)
                x_pixel = x_pixel.transpose(1, 2).reshape(B, N_q, C)
                x_pixel = self.out_proj_pixel(x_pixel)
                x_pixel = self.out_drop_pixel(x_pixel)
                
                x = x + self.pixel_adapter_ratio * x_pixel
            elif self.adapter_type == "serial":
                q_pixel = self.q_proj_pixel(x).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                kv_pixel = self.kv_proj_pixel(pixel_feats).reshape(B, N_i, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                k_pixel, v_pixel = kv_pixel.unbind(0)
                x = F.scaled_dot_product_attention(q_pixel, k_pixel, v_pixel, dropout_p=self.attn_drop.p if self.training else 0.0)
                x = x.transpose(1, 2).reshape(B, N_q, C)
                x = self.out_proj_pixel(x)
                x = self.out_drop_pixel(x)
            else:
                raise ValueError(f"Invalid adapter type: {self.adapter_type}")                                                               
        return x

        