import os

import numpy as np
import open3d as o3d
import trimesh

from src.utils.pcds.pcds import (
    axis_blender_to_opengl,
    read_pcd_from_ply,
    save_pcd_to_ply,
)


POINT_CLOUD_EXTS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
MESH_EXTS = {".obj", ".off", ".stl", ".glb", ".gltf", ".fbx"}


def infer_input_type(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in POINT_CLOUD_EXTS:
        return "point_cloud"
    if ext in MESH_EXTS:
        return "mesh"
    raise ValueError(f"Unsupported input format: {input_path}")


def _resample_point_cloud(points, num_points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape [N, 3], got {points.shape}")
    if len(points) == 0:
        raise ValueError("Input point cloud is empty")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    replace = len(points) < num_points
    indices = np.random.choice(len(points), num_points, replace=replace)
    return points[indices]


def load_point_cloud(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".ply":
        points, _ = read_pcd_from_ply(input_path)
        return points.astype(np.float32)

    point_cloud = o3d.io.read_point_cloud(input_path)
    if not point_cloud.has_points():
        raise ValueError(f"Failed to read point cloud from {input_path}")
    return np.asarray(point_cloud.points, dtype=np.float32)


def load_trimesh_mesh(input_path):
    mesh = trimesh.load(input_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No valid mesh geometry found in {input_path}")
        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh from {input_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {input_path}")
    return mesh


def sample_points_from_mesh(mesh, num_points):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces")

    mesh = mesh.copy()
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


def process_input(input_path, num_pcds=15000, blender_axis=False, output_dir=None):
    input_type = infer_input_type(input_path)

    if input_type == "point_cloud":
        points = load_point_cloud(input_path)
        if blender_axis:
            points = axis_blender_to_opengl(points)
        points = _resample_point_cloud(points, num_pcds)
        if output_dir is not None:
            save_pcd_to_ply(points, os.path.join(output_dir, "input_points_resampled.ply"))
        return points, input_type

    mesh = load_trimesh_mesh(input_path)
    points = sample_points_from_mesh(mesh, num_points=num_pcds)
    if blender_axis:
        points = axis_blender_to_opengl(points)
    if output_dir is not None:
        save_pcd_to_ply(points, os.path.join(output_dir, "input_points_sampled.ply"))
    return points, input_type
