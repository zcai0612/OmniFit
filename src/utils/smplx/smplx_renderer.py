import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
import smplx
import json

from matplotlib import cm as mpl_cm, colors as mpl_colors
from src.utils.mesh.render import Renderer
from src.utils.mesh.camera import Camera
from kiui.mesh import Mesh


def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
    vertex_labels = np.zeros(n_vertices)
    for part_idx, (k, v) in enumerate(part_segm.items()):
        vertex_labels[v]= part_idx
    cm = mpl_cm.get_cmap('jet')
    norm_gt = mpl_colors.Normalize()
    vertex_colors = np.ones((n_vertices, 4))
    vertex_colors[:, 3]= alpha
    vertex_colors[:,:3]= cm(norm_gt(vertex_labels))[:, :3]
    return vertex_colors

class SMPLXRenderer:
    def __init__(
        self,
        device: str = "cuda",
        part_segmentation_path: str = "human_models/smplx_vert_segmentation.json",
        background_color: str = "white",
    ):
        self.device = device
        with open(part_segmentation_path, "r") as f:
            self.part_segmentation = json.load(f)
        self.renderer = Renderer(device=device)
        self.camera = Camera(device=device)
        self.mvps, self.rots, _, _ = self.camera.get_orthogonal_camera([0])
        self.bg_color = self.renderer.get_bg_color(background_color).to(self.device)

    def render_front_view(self, mesh, height=1024, width=1024):
        render_pkg = self.renderer(
            mesh, mvp=self.mvps, h=height, w=width, shading_mode="lambertian", bg_color=self.bg_color
        )

        render_img = render_pkg["image"]
        alpha = render_pkg["alpha"]
        render_img_rgba = torch.cat([render_img, alpha], dim=-1)
        return render_img_rgba.permute(0, 3, 1, 2)
        

    def render_smplx_semantic(
        self, smplx_v, smplx_f, height=1024, width=1024
    ):
        smplx_v = torch.from_numpy(smplx_v).float().to(self.device)
        smplx_f = torch.from_numpy(smplx_f).int().to(self.device)
        vertex_colors = part_segm_to_vertex_colors(self.part_segmentation, smplx_v.shape[0])
        vertex_colors = torch.from_numpy(vertex_colors).float().to(self.device)
        mesh = Mesh(v=smplx_v, f=smplx_f, vc=vertex_colors, device=self.device)
        
        return self.render_front_view(mesh, height=height, width=width)
    
    def render_smplx_color(
        self, smplx_v, smplx_f, smplx_vc, height=1024, width=1024
    ):
        
        smplx_v = torch.from_numpy(smplx_v).float().to(self.device)
        smplx_f = torch.from_numpy(smplx_f).int().to(self.device)
        smplx_vc = torch.from_numpy(smplx_vc).float().to(self.device)
        mesh = Mesh(v=smplx_v, f=smplx_f, vc=smplx_vc, device=self.device)
        mesh.auto_normal()
        
        return self.render_front_view(mesh, height=height, width=width)
