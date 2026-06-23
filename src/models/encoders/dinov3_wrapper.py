import torch
import torch.nn as nn
import timm
from torchvision.transforms import Resize
from einops import rearrange
from transformers import AutoModel, AutoImageProcessor

class DINOv3Wrapper(nn.Module):
    def __init__(
        self,
        device="cuda",
        dtype=torch.float16,
        image_size=512,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self.model = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.model.to(device, dtype=dtype)
        self.model.eval()  # set to eval mode
        # DINOv2 preprocessing (ImageNet normalization)
        self.vit_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype)[:, None, None]
        self.vit_std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype)[:, None, None]

        self.image_size = image_size
        if self.image_size is not None:
            self.resize_transform = Resize((self.image_size, self.image_size))
            
    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        image_input = image.to(self.device, dtype=self.dtype)
        if self.image_size is not None:
            image_input = self.resize_transform(image_input)
        image_input = (image_input - self.vit_mean) / self.vit_std

        outputs = self.model.forward_features(image_input)
        # print(outputs.shape)
        features = outputs[:, 5:, :]  # Exclude CLS token and Registry token

        return features

# class DINOv3Wrapper(nn.Module):
#     def __init__(
#         self,
#         device="cuda",
#         dtype=torch.float16,
#         local_model_dir=None,
#         image_size=None,
#     ):
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         self.image_size = image_size
#         if local_model_dir is not None:
#             self.model = AutoModel.from_pretrained(
#                 'facebook/dinov3-vitl16-pretrain-lvd1689m',
#                 cache_dir=local_model_dir,
#             )
#         else:
#             self.model = AutoModel.from_pretrained(
#                 'facebook/dinov3-vitl16-pretrain-lvd1689m',
#             )
#         self.model.to(device, dtype=dtype)
#         self.model.eval()  # set to eval mode
#         # DINOv2 preprocessing (ImageNet normalization)
#         self.vit_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype)[:, None, None]
#         self.vit_std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype)[:, None, None]

#         self.image_size = image_size
#         if self.image_size is not None:
#             self.resize_transform = Resize((self.image_size, self.image_size))
            
#     @torch.no_grad()
#     def forward(self, image: torch.Tensor):
#         image_input = image.to(self.device, dtype=self.dtype)
#         if self.image_size is not None:
#             image_input = self.resize_transform(image_input)
#         image_input = (image_input - self.vit_mean) / self.vit_std

#         outputs = self.model(image_input, output_hidden_states=True)
#         hidden = outputs.hidden_states[-1]
#         features = hidden[:, 5:, :]  # Exclude CLS token and Registry token

#         return features