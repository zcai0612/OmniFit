import os
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import sys
from kiui.mesh import Mesh
from .render import Renderer
from .camera import Camera
from .mesh_util import normalize_vertices

class CommonRenderer:
    def __init__(
        self,
        device: str = "cuda",
        shading_mode: str = "albedo",
        background_color: str = "white",
    ):
        self.camera = Camera(device=device)
        self.device = device

        self.shading_mode = shading_mode
        self.renderer = Renderer(device=device)
        self.mvps, self.rots, _, _ = self.camera.get_orthogonal_camera([0])
        self.bg_color = self.renderer.get_bg_color(background_color).to(self.device)

    def render_mesh_color(
        self,
        mesh: Mesh,
        height: int = 1024,
        width: int = 1024,
    ):
        render_pkg = self.renderer(
            mesh, mvp=self.mvps, h=height, w=width, shading_mode="albedo", bg_color=self.bg_color
        )

        render_img = render_pkg["image"]
        alpha = render_pkg["alpha"]
        if render_img is None:
            raise ValueError("Mesh has no texture or vertex color, so it cannot be rendered as an RGB image.")
        render_img_rgba = torch.cat([render_img, alpha], dim=-1)
        return render_img_rgba.permute(0, 3, 1, 2)
    
    def render_mesh(
        self, mesh_path: str, height: int = 1024, width: int = 1024
    ):
        mesh = Mesh.load(mesh_path, resize=False)
        mesh.v = normalize_vertices(mesh.v, bound=0.9)
        return self.render_mesh_color(mesh, height, width)
