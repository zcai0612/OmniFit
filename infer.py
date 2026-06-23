import json
import argparse
import pickle as pkl
import os

import numpy as np
import torch
from PIL import Image
from smplx import SMPLX

from src.models.encoders.dinov2_wrapper import Dinov2Wrapper
from src.models.encoders.pointbert.custom_point_encoder import PointTransformer
from src.models.lmk_predictor import LmkPredictor
from src.models.scale_predictor import ScalePredictor
from src.utils.mesh.io import (
    infer_input_type,
    load_trimesh_mesh,
    process_input,
    sample_points_from_mesh,
)
from src.utils.mesh.mesh_util import center_vertices, normalize_vertices
from src.utils.pcds.pcds import (
    axis_blender_to_opengl,
    save_pcd_to_ply,
)
from src.utils.smplx.fitting_smplx import fit_smplx


def parse_args():
    parser = argparse.ArgumentParser(description="Infer SMPL-X from mesh or point cloud input.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input mesh or point cloud.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--num_points", type=int, default=15000, help="Number of points used for inference.")
    parser.add_argument("--num_betas", type=int, default=10, help="Number of SMPL-X betas.")
    parser.add_argument(
        "--point_encoder_ckpt",
        type=str,
        default="./weights/all_in_one/point_encoder.pt",
        help="Checkpoint path for point encoder.",
    )
    parser.add_argument(
        "--lmk_predictor_ckpt",
        type=str,
        default="./weights/all_in_one/lmk_predictor.pt",
        help="Checkpoint path for landmark predictor.",
    )
    parser.add_argument(
        "--scale_predictor_ckpt",
        type=str,
        default="./weights/all_in_one/scale_predictor.pt",
        help="Checkpoint path for scale predictor.",
    )
    parser.add_argument(
        "--lmk_json_file",
        type=str,
        default="data/smplx_600_landmark_253.json",
        help="SMPL-X landmark definition json.",
    )
    parser.add_argument(
        "--blender_axis",
        action="store_true",
        help="Convert coordinates from Blender axis to OpenGL axis.",
    )
    parser.add_argument(
        "--with_scale",
        action="store_true",
        help="Enable scale prediction before landmark inference.",
    )
    parser.add_argument(
        "--with_image_adapter",
        action="store_true",
        help="Enable image adapter for landmark prediction.",
    )
    parser.add_argument(
        "--image_adapter_ckpt",
        type=str,
        default="./weights/all_in_one/pixel_adapter.pt",
        help="Checkpoint path for image adapter.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path for image adapter. If omitted for mesh input, render a front-view image.",
    )
    return parser.parse_args()


def load_image_tensor(image_path, height=518, width=518):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize((width, height))
    image = torch.from_numpy(np.asarray(image)).float() / 255.0

    alpha = image[:, :, 3:4]
    image = image[:, :, :3] * alpha + (1.0 - alpha)
    return image.permute(2, 0, 1).unsqueeze(0)


def _rgba_to_rgb_tensor(image_tensor):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.shape[1] == 3:
        return image_tensor
    if image_tensor.shape[1] != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 channels, got {image_tensor.shape}")

    alpha = image_tensor[:, 3:4]
    return image_tensor[:, :3] * alpha + torch.ones_like(image_tensor[:, :3]) * (1.0 - alpha)


def save_image_tensor(image_tensor, output_path):
    image = _rgba_to_rgb_tensor(image_tensor).squeeze(0).detach().cpu().clamp(0.0, 1.0)
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(output_path)


def render_mesh_image_tensor(mesh_path, device, output_path=None, height=518, width=518):
    from src.utils.mesh.common_renderer import CommonRenderer

    renderer = CommonRenderer(device=device, background_color="white")
    image = renderer.render_mesh(mesh_path, height=height, width=width)
    if output_path is not None:
        save_image_tensor(image, output_path)
    return _rgba_to_rgb_tensor(image)


def load_model(
    device,
    point_encoder_ckpt,
    lmk_predictor_ckpt,
    scale_predictor_ckpt,
    image_adapter_ckpt=None,
    with_image_adapter=False,
    num_betas=10,
):
    if with_image_adapter and image_adapter_ckpt is None:
        raise ValueError("image_adapter_ckpt is required when with_image_adapter=True")

    point_encoder = PointTransformer(
        trans_dim=516,
        depth=16,
        drop_path_rate=0.1,
        num_heads=12,
        group_size=32,
        num_group=768,
        encoder_dims=512,
    )
    point_encoder.load_state_dict(torch.load(point_encoder_ckpt, map_location=device, weights_only=True))
    point_encoder.to(device)
    point_encoder.eval()

    lmk_predictor = LmkPredictor(
        num_lmks=600,
        embed_dim=512,
        depth=24,
        point_feat_dim=516,
        use_pixel_adapter=with_image_adapter,
    )
    lmk_predictor.load_weights(
        lmk_predictor_weight_path=lmk_predictor_ckpt,
        pixel_adapter_weight_path=image_adapter_ckpt if with_image_adapter else None,
    )
    lmk_predictor.to(device)
    lmk_predictor.eval()

    image_encoder = None
    if with_image_adapter:
        image_encoder = Dinov2Wrapper(
            device=device,
            image_size=518,
        )

    scale_predictor = ScalePredictor(
        trans_dim=516,
        depth=12,
        num_heads=12,
        group_size=32,
        num_group=768,
        encoder_dims=512,
        head_hidden_dim=256,
        head_num_layers=2,
    )
    scale_predictor.load_state_dict(torch.load(scale_predictor_ckpt, map_location=device, weights_only=True))
    scale_predictor.to(device)
    scale_predictor.eval()

    smplx_model = SMPLX(
        model_path="human_models/models/smplx/SMPLX_NEUTRAL.npz",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=num_betas,
    )
    smplx_model = smplx_model.to(device)

    return point_encoder, lmk_predictor, image_encoder, scale_predictor, smplx_model


def infer_case(
    device,
    input_path,
    point_encoder,
    lmk_predictor,
    image_encoder,
    smplx_model,
    scale_predictor,
    lmk_json_file,
    blender_axis=False,
    with_scale=True,
    with_image_adapter=False,
    image_path=None,
    num_points=15000,
    output_dir=None,
):
    os.makedirs(output_dir, exist_ok=True)

    input_type = infer_input_type(input_path)
    final_scale = 1.0
    adapter_mesh_path = None

    if input_type == "mesh":
        mesh = load_trimesh_mesh(input_path)
        if blender_axis:
            mesh.vertices = axis_blender_to_opengl(mesh.vertices).astype(np.float32)
            mesh.export(os.path.join(output_dir, "converted_mesh.obj"))

        if with_scale:
            mesh.vertices, _, scale = normalize_vertices(mesh.vertices, bound=0.9, return_params=True)
            mesh.export(os.path.join(output_dir, "normalized_mesh.obj"))
            outer_points = sample_points_from_mesh(mesh, num_points=num_points)
            points_tensor = torch.from_numpy(outer_points).unsqueeze(0).to(device).float()

            with torch.no_grad():
                pred_scale = scale_predictor(points_tensor)[0].item()

            final_scale = pred_scale * scale
            mesh.vertices = (mesh.vertices * pred_scale).astype(np.float32)
            mesh.export(os.path.join(output_dir, "rescaled_mesh.obj"))
            outer_points = sample_points_from_mesh(mesh, num_points=num_points)
        else:
            outer_points = sample_points_from_mesh(mesh, num_points=num_points)

        save_pcd_to_ply(outer_points, os.path.join(output_dir, "input_points_sampled.ply"))

        if with_image_adapter and image_path is None:
            adapter_mesh_path = os.path.join(output_dir, "image_adapter_front_mesh.obj")
            mesh.export(adapter_mesh_path)
    else:
        outer_points, _ = process_input(
            input_path=input_path,
            num_pcds=num_points,
            blender_axis=blender_axis,
            output_dir=output_dir,
        )
        if with_scale:
            outer_points, _, scale = normalize_vertices(outer_points, bound=0.9, return_params=True)
            points_tensor = torch.from_numpy(outer_points).unsqueeze(0).to(device).float()

            with torch.no_grad():
                pred_scale = scale_predictor(points_tensor)[0].item()
                
            final_scale = pred_scale * scale
            outer_points = (outer_points * pred_scale).astype(np.float32)
            save_pcd_to_ply(outer_points, os.path.join(output_dir, "rescaled_points.ply"))

    image_tensor = None
    image_source = None
    if with_image_adapter:
        if image_encoder is None:
            raise ValueError("image_encoder is required when with_image_adapter=True")

        if image_path is not None:
            image_tensor = load_image_tensor(image_path).to(device)
            image_source = image_path
        else:
            if adapter_mesh_path is None:
                raise ValueError("--image is required for point cloud inputs when --with_image_adapter is enabled")
            image_source = os.path.join(output_dir, "image_adapter_front.png")
            image_tensor = render_mesh_image_tensor(
                adapter_mesh_path,
                device=device,
                output_path=image_source,
            ).to(device)

    outer_points_tensor = torch.from_numpy(outer_points).float().to(device)
    outer_points_tensor, center = center_vertices(outer_points_tensor, return_params=True)
    outer_points_tensor = outer_points_tensor.unsqueeze(0)

    with torch.no_grad():
        point_feats = point_encoder(outer_points_tensor)
        image_feats = image_encoder(image_tensor) if with_image_adapter else None

        pred_lmks = lmk_predictor(point_feats, image_feats)
        pred_lmks = pred_lmks + center.unsqueeze(0)

        fitted_smplx_mesh_list, fitted_smplx_params = fit_smplx(
            smplx_model,
            pred_lmks.to(torch.float32),
            lmk_json_file=lmk_json_file,
        )
        
    fitted_smplx_mesh = fitted_smplx_mesh_list[0]
    pred_lmks = pred_lmks.squeeze(0).detach().cpu().numpy()
    save_pcd_to_ply(pred_lmks, os.path.join(output_dir, "pred_lmks.ply"))
    fitted_smplx_mesh.export(os.path.join(output_dir, "fitted_smplx_mesh.obj"))

    with open(os.path.join(output_dir, "fitted_smplx_params.pkl"), "wb") as f:
        pkl.dump(fitted_smplx_params, f)

    def _to_serializable(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().tolist()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    fitted_smplx_params_json = {k: _to_serializable(v) for k, v in fitted_smplx_params.items()}
    fitted_smplx_params_json["scale"] = float(final_scale)
    fitted_smplx_params_json["input_type"] = input_type
    fitted_smplx_params_json["with_image_adapter"] = bool(with_image_adapter)
    fitted_smplx_params_json["image_source"] = image_source
    with open(os.path.join(output_dir, "fitted_smplx_params.json"), "w") as f:
        json.dump(fitted_smplx_params_json, f)


if __name__ == "__main__":
    args = parse_args()

    point_encoder, lmk_predictor, image_encoder, scale_predictor, smplx_model = load_model(
        device=args.device,
        point_encoder_ckpt=args.point_encoder_ckpt,
        lmk_predictor_ckpt=args.lmk_predictor_ckpt,
        scale_predictor_ckpt=args.scale_predictor_ckpt,
        image_adapter_ckpt=args.image_adapter_ckpt,
        with_image_adapter=args.with_image_adapter,
        num_betas=args.num_betas,
    )

    infer_case(
        device=args.device,
        input_path=args.input_path,
        point_encoder=point_encoder,
        lmk_predictor=lmk_predictor,
        image_encoder=image_encoder,
        smplx_model=smplx_model,
        scale_predictor=scale_predictor,
        lmk_json_file=args.lmk_json_file,
        blender_axis=args.blender_axis,
        with_scale=args.with_scale,
        with_image_adapter=args.with_image_adapter,
        image_path=args.image,
        num_points=args.num_points,
        output_dir=args.output_dir,
    )
