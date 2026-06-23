import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Resize
from einops import rearrange

def load_model(checkpoint, use_torchscript=False, device='cuda'):
    if use_torchscript:
        return torch.jit.load(checkpoint, map_location=device)
    else:
        return torch.export.load(checkpoint).module()

class SapiensWrapper(nn.Module):
    def __init__(
        self,
        model_name,
        device='cuda',
        dtype=torch.float16, 
        freeze: bool = True,
        image_size=1024,
    ):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.model = self._build_sapiens(model_name, device=device).to(device, dtype=dtype)
        
        if freeze:
            self._freeze()
        
        self.mean = torch.tensor([0.4844, 0.4570, 0.4062], device=self.device, dtype=self.dtype)[:, None, None]
        self.std = torch.Tensor([0.2295, 0.2236, 0.2256]).to(self.device, dtype=self.dtype)[:, None, None]
        
        self.image_size = image_size
        if self.image_size is not None:
            self.resize_transform = Resize((self.image_size, self.image_size))
        
        
    def _build_sapiens(self, model_name: str, device='cuda'):
        USE_TORCHSCRIPT = "_torchscript" in model_name
        
        model = load_model(model_name, use_torchscript=USE_TORCHSCRIPT, device=device)
        if not USE_TORCHSCRIPT:
            raise NotImplementedError
        return model
        
    
    def _freeze(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        image_input = image.to(self.device, dtype=self.dtype)
        if self.image_size is not None:
            image_input = self.resize_transform(image_input)
        image_input = (image_input - self.mean) / self.std
        
        (feats,) = self.model(image_input)
        feats = rearrange(feats, 'b c h w -> b (h w) c')
        return feats.float()
        