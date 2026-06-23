import numpy as np 
import torch
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


def downsample_pcds_batch_tensor(pcds_batch, num_points, method='random', return_indices=False):   
    """
    Downsample a batch of point cloud data to a fixed number of points
    
    Args:
        pcds_batch: Batch of point cloud data (torch tensor of shape [B, N, 3])
        num_points: Number of points to downsample to
        method: Downsampling method - 'random' (fast) or 'fps' (farthest point sampling, slower but more uniform)
        return_indices: If True, also return the indices of selected points (default: False)
    Returns:
        if return_indices is False:
            downsampled_pcds_batch: Downsampled batch of point cloud data (torch tensor of shape [B, num_points, 3])
        if return_indices is True:
            (downsampled_pcds_batch, indices): Tuple of downsampled points and their indices (torch tensor of shape [B, num_points])
    """
    B, N, C = pcds_batch.shape
    device = pcds_batch.device
    
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    if num_points >= N:
        # If requested points >= available points, return all points
        if return_indices:
            indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return pcds_batch, indices
        return pcds_batch
    
    if method == 'random':
        # Random sampling - fastest method
        # Generate random indices for each batch
        indices = torch.randperm(N, device=device)[:num_points].unsqueeze(0).expand(B, -1)
        
        # Alternative: different random sampling for each batch
        # indices = torch.stack([torch.randperm(N, device=device)[:num_points] for _ in range(B)])
        
        # Gather points
        downsampled = torch.gather(pcds_batch, 1, indices.unsqueeze(-1).expand(-1, -1, C))
        
        if return_indices:
            return downsampled, indices
        return downsampled
    
    elif method == 'fps':
        # Farthest Point Sampling - more uniform distribution but slower
        return farthest_point_sampling_batch(pcds_batch, num_points, return_indices=return_indices)
    
    else:
        raise ValueError(f"Unknown downsampling method: {method}. Choose 'random' or 'fps'")


def farthest_point_sampling_batch(points, num_samples, return_indices=False):
    """
    Farthest Point Sampling (FPS) for batch of point clouds - vectorized implementation
    
    Args:
        points: Batch of point cloud data (torch tensor of shape [B, N, 3])
        num_samples: Number of points to sample
        return_indices: If True, also return the indices of selected points (default: False)
    Returns:
        if return_indices is False:
            sampled_points: Sampled points (torch tensor of shape [B, num_samples, 3])
        if return_indices is True:
            (sampled_points, indices): Tuple of sampled points and their indices (torch tensor of shape [B, num_samples])
    """
    device = points.device
    B, N, C = points.shape
    
    if num_samples >= N:
        if return_indices:
            indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return points, indices
        return points
    
    # Initialize
    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Start with a random point for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(num_samples):
        # Record the current farthest point
        centroids[:, i] = farthest
        
        # Get the coordinates of current farthest point
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        
        # Calculate distance to current centroid
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        
        # Update distances: keep minimum distance to any selected point
        distance = torch.min(distance, dist)
        
        # Select the farthest point
        farthest = torch.max(distance, dim=-1)[1]
    
    # Gather the sampled points
    centroids_expanded = centroids.unsqueeze(-1).expand(-1, -1, C)
    sampled_points = torch.gather(points, 1, centroids_expanded)
    
    if return_indices:
        return sampled_points, centroids
    return sampled_points


def downsample_pcds(pcds, num_points):
    """
    Downsample point cloud data to a fixed number of points
    
    Args:
        pcds: Point cloud data (numpy array of shape [N, 3])
        num_points: Number of points to downsample to
    Returns:
        downsampled_pcds: Downsampled point cloud data (numpy array of shape [num_points, 3])
    """
    pcds = np.asarray(pcds, dtype=np.float32)
    
    if len(pcds) == 0:
        raise ValueError("Input point cloud is empty")
    
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    if num_points >= len(pcds):
        # If requested points >= available points, return all points
        return pcds
    
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds)
    
    # Use uniform downsampling to get approximately num_points
    # Calculate the downsampling factor
    downsample_factor = max(1, len(pcds) // num_points)
    downsampled_pcd = pcd.uniform_down_sample(downsample_factor)
    
    downsampled_points = np.asarray(downsampled_pcd.points, dtype=np.float32)
    
    # If we have too many or too few points after uniform downsampling, 
    # use random sampling to get exact number
    current_num = len(downsampled_points)
    if current_num != num_points:
        if current_num > num_points:
            # Random sampling without replacement
            indices = np.random.choice(current_num, num_points, replace=False)
            downsampled_points = downsampled_points[indices]
        else:
            # Random sampling with replacement (need more points than available)
            indices = np.random.choice(current_num, num_points, replace=True)
            downsampled_points = downsampled_points[indices]
    
    return downsampled_points

def upsample_pcds(pcds, num_points):
    """
    Upsample point cloud data to a fixed number of points while preserving surface characteristics
    
    Args:
        pcds: Point cloud data (numpy array of shape [N, 3])
        num_points: Number of points to upsample to
    Returns:
        upsampled_pcds: Upsampled point cloud data (numpy array of shape [num_points, 3])
    """
    pcds = np.asarray(pcds, dtype=np.float32)
    
    if len(pcds) == 0:
        raise ValueError("Input point cloud is empty")
    
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    if num_points <= len(pcds):
        # If requested points <= available points, return original or downsample
        return pcds if num_points == len(pcds) else downsample_pcds(pcds, num_points)
    
    # Calculate how many additional points we need
    additional_points_needed = num_points - len(pcds)
    
    # Randomly select base points for interpolation
    base_indices = np.random.choice(len(pcds), additional_points_needed, replace=True)
    
    # For each base point, find a nearby point and interpolate
    # Use vectorized operations for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(pcds)
    
    # Query k nearest neighbors for all base points at once
    k = min(4, len(pcds))
    distances, neighbor_indices = tree.query(pcds[base_indices], k=k)
    
    # For each base point, randomly select one of its neighbors (excluding itself)
    new_points = []
    for i in range(additional_points_needed):
        base_point = pcds[base_indices[i]]
        
        # Select a random neighbor (skip first one if it's the point itself)
        start_idx = 1 if distances[i, 0] < 1e-6 else 0
        if start_idx < len(neighbor_indices[i]):
            neighbor_idx = np.random.randint(start_idx, min(start_idx + 3, len(neighbor_indices[i])))
            neighbor_point = pcds[neighbor_indices[i, neighbor_idx]]
            
            # Interpolate with random weight biased towards surface preservation
            alpha = np.random.uniform(0.2, 0.5)
            new_point = (1 - alpha) * base_point + alpha * neighbor_point
        else:
            # Fallback: use base point with tiny noise
            new_point = base_point + np.random.normal(0, 0.001, 3)
        
        new_points.append(new_point)
    
    # Combine original and new points
    new_points = np.array(new_points, dtype=np.float32)
    upsampled_pcds = np.vstack([pcds, new_points])
    
    return upsampled_pcds


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for uniform point distribution
    
    Args:
        points: Point cloud data (numpy array of shape [N, 3])
        num_samples: Number of points to sample
    Returns:
        indices: Indices of sampled points (numpy array of shape [num_samples,])
    """
    points = np.asarray(points, dtype=np.float32)
    N = len(points)
    
    if num_samples >= N:
        return np.arange(N)
    
    # Initialize
    indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf, dtype=np.float32)
    
    # Start with a random point
    current_idx = np.random.randint(0, N)
    
    for i in range(num_samples):
        indices[i] = current_idx
        current_point = points[current_idx]
        
        # Update distances to the farthest point
        dist_to_current = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist_to_current)
        
        # Select the farthest point
        current_idx = np.argmax(distances)
    
    return indices


def sample_points(obj_path, num_points=20000, remove_float=False):
    """
    sample points uniformly on the surface of a 3D mesh
    Args:
        obj_path: path to the OBJ file
        num_points: number of points to sample
        remove_float: whether to remove floating points (default: True)
    Returns:
        points: sampled point cloud, a numpy array of shape (num_points, 3)
    """
    try:
        # Load the OBJ file with error handling
        loaded = trimesh.load(obj_path)
        mesh = loaded
        
        # Validate mesh
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid Trimesh: {type(mesh)}")
        
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
        
        # Remove floating/outlier components if requested
        if remove_float:
            try:
                # Split mesh into connected components with safety checks
                components = mesh.split(only_watertight=False)
                
                if len(components) > 1:
                    # Filter out empty or invalid components
                    valid_components = [comp for comp in components 
                                      if len(comp.vertices) > 0 and len(comp.faces) > 0]
                    
                    if not valid_components:
                        print("Warning: No valid components found after splitting, using original mesh")
                    else:
                        # Find the largest component by face count
                        largest_component = max(valid_components, key=lambda x: len(x.faces))
                        
                        # Validate the largest component
                        if len(largest_component.faces) > 0:
                            mesh = largest_component
                        
                        # Optional: Also remove small components by area threshold
                        # This helps remove tiny floating pieces that might still be connected
                        try:
                            total_area = mesh.area
                            if total_area > 0:
                                min_area_threshold = total_area * 0.01  # Remove components smaller than 1% of total area
                                
                                # Re-split and filter by area
                                components = mesh.split(only_watertight=False)
                                if len(components) > 1:
                                    filtered_components = [comp for comp in components 
                                                         if comp.area >= min_area_threshold]
                                    if filtered_components:
                                        # Merge remaining components back together
                                        mesh = trimesh.util.concatenate(filtered_components)
                        except Exception as e:
                            print(f"Warning: Area-based filtering failed: {e}, continuing with largest component")
            
            except Exception as e:
                print(f"Warning: Component removal failed: {e}, using original mesh")
        
        # Final validation before sampling
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Final mesh has no vertices or faces")
        
        # Check mesh area
        try:
            mesh_area = mesh.area
            if mesh_area <= 0:
                raise ValueError("Mesh has zero or negative area")
        except Exception as e:
            print(f"Warning: Could not compute mesh area: {e}")
        
        # Sample points uniformly on the surface of the mesh
        try:
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
        except Exception as e:
            print(f"Error during surface sampling: {e}")
            # Fallback: try with a simpler approach
            try:
                # Try with even sampling
                points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
            except Exception as e2:
                print(f"Even sampling also failed: {e2}")
                # Last resort: sample from vertices
                if len(mesh.vertices) >= num_points:
                    indices = np.random.choice(len(mesh.vertices), num_points, replace=False)
                    points = mesh.vertices[indices]
                else:
                    # If not enough vertices, sample with replacement
                    indices = np.random.choice(len(mesh.vertices), num_points, replace=True)
                    points = mesh.vertices[indices]
        
        # Validate output
        if points is None or len(points) == 0:
            raise ValueError("Failed to sample any points")
        
        # Convert the sampled points to a numpy array and return
        return points.astype(np.float32)
    
    except Exception as e:
        print(f"Error in sample_points: {e}")
        raise

def save_pcd_to_ply(pcd, output_path, colors=None):
    """
    Save point cloud data to PLY format file
    
    Args:
        pcd: Point cloud data (numpy array of shape [N, 3] or open3d PointCloud object)
        output_path: Output file path with .ply extension
        colors: Optional colors for each point (numpy array of shape [N, 3])
                Color values should be in range [0, 1] for RGB
    Returns:
        success: Boolean indicating if the save was successful
    """
    if isinstance(pcd, np.ndarray):
        # Convert numpy array to open3d point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        
        # Add colors if provided
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.shape[0] != pcd.shape[0]:
                raise ValueError(f"Number of colors ({colors.shape[0]}) must match number of points ({pcd.shape[0]})")
            if colors.shape[1] != 3:
                raise ValueError(f"Colors must have shape [N, 3], got {colors.shape}")
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
    elif hasattr(pcd, 'points'):
        # Already an open3d point cloud object
        point_cloud = pcd
        
        # Override colors if provided
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            num_points = len(np.asarray(point_cloud.points))
            if colors.shape[0] != num_points:
                raise ValueError(f"Number of colors ({colors.shape[0]}) must match number of points ({num_points})")
            if colors.shape[1] != 3:
                raise ValueError(f"Colors must have shape [N, 3], got {colors.shape}")
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise ValueError("Unsupported point cloud data type")
    
    # Save to PLY file
    success = o3d.io.write_point_cloud(output_path, point_cloud)
    
    if not success:
        raise RuntimeError(f"Failed to save point cloud to {output_path}")
    
    return success

def read_pcd_with_color_from_ply(ply_path):
    """
    Read point cloud data with color from PLY format file
    
    Args:
        ply_path: Input file path with .ply extension
    Returns:
        pcd: Point cloud data (numpy array of shape [N, 3])
        colors: Point cloud colors (numpy array of shape [N, 3])
    """
    # Read PLY file
    point_cloud = o3d.io.read_point_cloud(ply_path)
    
    if not point_cloud.has_points():
        raise ValueError(f"Failed to read point cloud from {ply_path}")
    
    points = np.asarray(point_cloud.points)
    
    # Check if the point cloud has colors
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        return points, colors
    else:
        raise ValueError(f"Point cloud from {ply_path} does not have color information")


def read_pcd_from_ply(ply_path):
    """
    Read point cloud data from PLY format file
    
    Args:
        ply_path: Input file path with .ply extension
    Returns:
        pcd: Point cloud data (numpy array of shape [N, 3])
        normals: Point cloud normals (numpy array of shape [N, 3]) if available, otherwise None
    """
    # Read PLY file
    point_cloud = o3d.io.read_point_cloud(ply_path)
    
    if not point_cloud.has_points():
        raise ValueError(f"Failed to read point cloud from {ply_path}")
    
    points = np.asarray(point_cloud.points)
    
    # Check if the point cloud has normals
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        return points, normals
    else:
        return points, None


def pcds_to_colored_ball_mesh(pcds, colors, radius=0.01):
    """
    Convert point cloud data to a colored ball mesh
    
    Args:
        pcds: Point cloud data (numpy array of shape [N, 3])
        colors: Colors for each point (numpy array of shape [N, 3])
        radius: Radius of the balls
    Returns:
        mesh: Colored ball mesh (open3d TriangleMesh object)
    """
    # Create a sphere mesh for each point
    meshes = []
    for pcd, color in zip(pcds, colors):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        ball.paint_uniform_color(color)
        ball.translate(pcd)
        meshes.append(ball)
    
    # Combine all meshes
    mesh = o3d.geometry.TriangleMesh()
    for ball in meshes:
        mesh += ball
    
    return mesh


def save_vector_to_ply(
    points_start, points_end, output_path, 
    colors_start=[1.0, 0.0, 0.0], colors_end=[0.0, 1.0, 0.0], colors_line=[0.0, 0.0, 1.0]
):
    """
    Save vector data to PLY format file - combines start points, end points, and vectors in one file
    
    Args:
        points_start: Start points of vectors (numpy array of shape [N, 3])
        points_end: End points of vectors (numpy array of shape [N, 3])
        output_path: Output file path with .ply extension
        colors_start: Colors for start points (numpy array of shape [N, 3] or list of 3 floats)
        colors_end: Colors for end points (numpy array of shape [N, 3] or list of 3 floats)
        colors_line: Colors for lines (numpy array of shape [N, 3] or list of 3 floats)
    """
    # Convert inputs to numpy arrays
    points_start = np.asarray(points_start, dtype=np.float32)
    points_end = np.asarray(points_end, dtype=np.float32)
    
    # Validate input shapes
    if points_start.shape != points_end.shape:
        raise ValueError("points_start and points_end must have the same shape")
    
    if len(points_start.shape) != 2 or points_start.shape[1] != 3:
        raise ValueError("Points must be of shape [N, 3]")
    
    num_points = points_start.shape[0]
    
    # Handle colors
    colors_start = np.asarray(colors_start, dtype=np.float32)
    colors_end = np.asarray(colors_end, dtype=np.float32)
    colors_line = np.asarray(colors_line, dtype=np.float32)
    
    # If colors are single color (shape [3,]), broadcast to all points
    if colors_start.shape == (3,):
        colors_start = np.tile(colors_start, (num_points, 1))
    elif colors_start.shape != (num_points, 3):
        raise ValueError("colors_start must be either [3,] or [N, 3] shape")
        
    if colors_end.shape == (3,):
        colors_end = np.tile(colors_end, (num_points, 1))
    elif colors_end.shape != (num_points, 3):
        raise ValueError("colors_end must be either [3,] or [N, 3] shape")
    
    if colors_line.shape == (3,):
        colors_line = np.tile(colors_line, (num_points, 1))
    elif colors_line.shape != (num_points, 3):
        raise ValueError("colors_line must be either [3,] or [N, 3] shape")
    
    # Combine all points (start points + end points)
    all_points = np.vstack([points_start, points_end])
    all_colors = np.vstack([colors_start, colors_end])
    
    # Create line indices (connecting each start point to its corresponding end point)
    lines = []
    for i in range(num_points):
        lines.append([i, i + num_points])  # Connect start point i to end point i
    
    lines = np.array(lines, dtype=np.int32)
    
    # Create Open3D LineSet geometry
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_line)  # Use dedicated line colors
    
    # Save to PLY file
    success = o3d.io.write_line_set(output_path, line_set)
    
    if not success:
        raise RuntimeError(f"Failed to save vector data to {output_path}")
    
    print(f"Successfully saved {num_points} vectors to {output_path}")
    return success

def axis_blender_to_opengl(pcds):
    """
    Convert point cloud data from Blender coordinate system (Z up) to OpenGL coordinate system (Y up)
    
    Args:
        pcds: Point cloud data in Blender coordinates (numpy array of shape [N, 3] or torch tensor of shape [N, 3])
    Returns:
        converted_pcds: Point cloud data in OpenGL coordinates (numpy array of shape [N, 3] or torch tensor of shape [N, 3])
    """
    # Swap Y and Z axes, and invert Z axis
    if isinstance(pcds, np.ndarray):
        converted_pcds = pcds.copy()
        converted_pcds[:, [1, 2]] = converted_pcds[:, [2, 1]]  # Swap Y and Z
        converted_pcds[:, 2] = -converted_pcds[:, 2]  # Invert new Z axis
        return converted_pcds
    elif isinstance(pcds, torch.Tensor):    
        converted_pcds = pcds.clone()
        converted_pcds[:, [1, 2]] = converted_pcds[:, [2, 1]]  # Swap Y and Z
        converted_pcds[:, 2] = -converted_pcds[:, 2]  # Invert new Z axis
    
    return converted_pcds