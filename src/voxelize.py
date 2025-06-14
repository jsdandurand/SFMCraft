import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

def load_point_cloud(ply_path):
    """
    Load a point cloud from a PLY file.
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
    
    print(f"Loading point cloud from {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in {ply_path}. The dense reconstruction might have failed.")
    
    print(f"Point cloud loaded with {len(pcd.points)} points")
    return pcd

def normalize_point_cloud(pcd, target_size=1.0):
    """
    Center and normalize the point cloud to fit in a cube of given size.
    """
    # Get the bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    scale = np.max(bbox.get_extent())
    
    # Center and scale
    pcd.translate(-center)
    pcd.scale(target_size / scale, center=[0, 0, 0])
    
    return pcd

def crop_to_center(pcd, crop_factor=0.8):
    """
    Crop the point cloud to keep only the central region around the mean point position.
    crop_factor: float between 0 and 1, representing how much of the total size to keep
                (e.g., 0.8 means keep the central 80% of the point cloud)
    """
    # Get points and compute mean position (true center of mass)
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    
    # Get the axis-aligned bounding box just for the extent
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    # Calculate new bounds based on crop factor and centered on mean position
    half_size = extent * (crop_factor / 2)
    min_bound = center - half_size
    max_bound = center + half_size
    
    # Create cropping box centered on mean position
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Crop the point cloud
    cropped_pcd = pcd.crop(crop_box)
    
    print(f"Original points: {len(pcd.points)}")
    print(f"Points after cropping: {len(cropped_pcd.points)}")
    print(f"Removed {len(pcd.points) - len(cropped_pcd.points)} points ({(1 - len(cropped_pcd.points)/len(pcd.points))*100:.1f}%)")
    print(f"Crop box center: {center}")
    print(f"Crop box size: {extent * crop_factor}")
    
    return cropped_pcd

def create_voxel_grid(pcd, voxel_size=0.05):
    """
    Convert point cloud to voxel grid.
    Returns both the voxel grid and voxel coordinates/colors for visualization.
    """
    print(f"Creating voxel grid with voxel size: {voxel_size}")
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    voxels = voxel_grid.get_voxels()
    
    if len(voxels) == 0:
        raise ValueError("No voxels created. Try adjusting the voxel size.")
    
    print(f"Voxel grid created with {len(voxels)} voxels")
    
    # Extract voxel coordinates and colors
    voxel_coords = np.array([voxel.grid_index for voxel in voxels], dtype=np.int32)
    voxel_colors = np.array([voxel.color if voxel.color is not None else [0.7, 0.7, 0.7] for voxel in voxels])
    
    # Find grid dimensions
    grid_dims = np.max(voxel_coords, axis=0) + 1
    
    # Create 4D grid: (x, y, z, channels) where channels are [occupancy, R, G, B]
    grid = np.zeros((*grid_dims, 4), dtype=np.float32)
    
    # Fill grid with occupancy and colors
    for coord, color in zip(voxel_coords, voxel_colors):
        x, y, z = coord
        grid[x, y, z, 0] = 1.0  # occupancy
        grid[x, y, z, 1:] = color  # RGB colors
    
    # Scale coordinates by voxel size for visualization
    voxel_coords = voxel_coords * voxel_size + np.asarray(voxel_grid.origin)
    
    return voxel_grid, voxel_coords, voxel_colors, grid

def save_voxel_grid(grid, output_path):
    """
    Save the voxel grid as a 4D array containing occupancy and colors.
    Grid shape is (x, y, z, 4) where the last dimension contains [occupancy, R, G, B]
    """
    # Ensure the output path ends with .npy
    if not output_path.endswith('.npy'):
        output_path = output_path.rsplit('.', 1)[0] + '.npy'
    
    # Save as numpy file
    np.save(output_path, grid)
    print(f"Voxel grid saved to {output_path}")
    print(f"Grid shape: {grid.shape}")

def save_visualization(pcd, voxel_coords, voxel_colors, output_path):
    """
    Save a visualization of the point cloud and voxel grid as images.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Create point cloud visualization
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pcd_vis.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    # Create voxel visualization as point cloud
    voxel_vis = o3d.geometry.PointCloud()
    voxel_vis.points = o3d.utility.Vector3dVector(voxel_coords)
    voxel_vis.colors = o3d.utility.Vector3dVector(voxel_colors)
    
    # Save original point cloud
    o3d.io.write_point_cloud(os.path.join(output_path, "point_cloud.ply"), pcd_vis)
    
    # Save voxelized version
    o3d.io.write_point_cloud(os.path.join(output_path, "voxels.ply"), voxel_vis)
    
    print(f"Visualizations saved to {output_path}")

def detect_floor_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=100):
    """Detect the floor plane and return its model parameters and inlier points.
    Returns:
        plane_model: [a, b, c, d] where ax + by + cz + d = 0 is the plane equation
        inliers: indices of points that belong to the floor
        no_floor_pcd: point cloud with floor points removed
    """
    print("\nDetecting floor plane...")
    
    # Segment plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Create mask for non-floor points (inverse of inliers)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    
    # Keep only non-floor points
    remaining_points = points[mask]
    remaining_colors = colors[mask]
    
    # Create new point cloud without the floor
    no_floor_pcd = o3d.geometry.PointCloud()
    no_floor_pcd.points = o3d.utility.Vector3dVector(remaining_points)
    no_floor_pcd.colors = o3d.utility.Vector3dVector(remaining_colors)
    
    print(f"Found floor plane with {len(inliers)} points")
    print(f"Remaining points: {len(remaining_points)}")
    
    return plane_model, inliers, no_floor_pcd

def orient_by_floor(pcd, plane_model):
    """Orient the point cloud using the floor plane normal vector.
    The floor normal will become the Y axis (up direction)."""
    print("\nOrienting point cloud using floor plane...")
    
    points = np.asarray(pcd.points)
    
    # Extract floor normal vector (a, b, c) from plane equation ax + by + cz + d = 0
    normal = plane_model[:3]
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Determine if normal points up or down
    if normal[1] < 0:  # If Y component is negative, flip it
        normal = -normal
    
    # Create rotation matrix that aligns floor normal with Y axis [0, 1, 0]
    # First create a vector perpendicular to normal (this will help form the new X axis)
    if abs(normal[0]) < abs(normal[2]):
        # If X component is smaller, use X axis for cross product
        perpendicular = np.cross(normal, [1, 0, 0])
    else:
        # If Z component is smaller, use Z axis for cross product
        perpendicular = np.cross(normal, [0, 0, 1])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Create third vector to complete orthogonal basis
    third = np.cross(normal, perpendicular)
    third = third / np.linalg.norm(third)
    
    # Create rotation matrix
    # The new Y axis is the normal, X axis is perpendicular, Z axis is third
    R = np.column_stack([perpendicular, normal, third])
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    
    # Center points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Apply rotation
    aligned = np.dot(centered, R)
    
    return aligned + centroid, R

def remove_outliers(pcd, nb_neighbors=10, std_ratio=1.0):
    """Remove outliers using statistical analysis with KNN.
    A point is considered an outlier if its average distance to its k nearest neighbors
    is above std_ratio times the standard deviation of the average distances."""
    print("\nRemoving outlier points...")
    cleaned_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Removed {len(pcd.points) - len(cleaned_pcd.points)} outlier points")
    return cleaned_pcd

def isolate_main_object(pcd, eps=0.01, min_points=100):
    """Isolate the main object (largest cluster) using DBSCAN clustering.
    eps: maximum distance between points in the same cluster
    min_points: minimum number of points to form a cluster"""
    print("\nIsolating main object using DBSCAN clustering...")
    
    # Run DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if len(np.unique(labels)) <= 1:
        print("Warning: No clusters found. Try adjusting eps and min_points parameters.")
        return pcd
    
    # Remove noise points (labeled as -1)
    n_noise = len(labels[labels == -1])
    print(f"Found {len(np.unique(labels)) - 1} clusters")
    print(f"Removed {n_noise} noise points")
    
    # Find the largest cluster
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
    largest_cluster = max(unique_labels, key=lambda x: np.sum(labels == x))
    
    # Keep only points in the largest cluster
    mask = labels == largest_cluster
    points = np.asarray(pcd.points)[mask]
    colors = np.asarray(pcd.colors)[mask]
    
    # Create new point cloud with only the largest cluster
    isolated_pcd = o3d.geometry.PointCloud()
    isolated_pcd.points = o3d.utility.Vector3dVector(points)
    isolated_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Kept largest cluster with {len(points)} points")
    print(f"Removed {len(pcd.points) - len(points)} points from other clusters")
    
    return isolated_pcd

def remove_floor(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=100):
    """Remove the floor by detecting the largest planar surface.
    
    Args:
        pcd: Open3D point cloud
        distance_threshold: Maximum distance a point can be from the plane to be considered part of it
        ransac_n: Number of points to sample for each RANSAC iteration
        num_iterations: Number of RANSAC iterations
    """
    print("\nDetecting and removing floor plane...")
    
    # Segment plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # Create mask for non-floor points (inverse of inliers)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    
    # Keep only non-floor points
    remaining_points = points[mask]
    remaining_colors = colors[mask]
    
    # Create new point cloud without the floor
    no_floor_pcd = o3d.geometry.PointCloud()
    no_floor_pcd.points = o3d.utility.Vector3dVector(remaining_points)
    no_floor_pcd.colors = o3d.utility.Vector3dVector(remaining_colors)
    
    print(f"Removed {len(inliers)} floor points")
    print(f"Remaining points: {len(remaining_points)}")
    
    # If too many points were removed (>80%), it might have detected the wrong plane
    if len(inliers) > 0.8 * len(points):
        print("Warning: Large portion of points were removed. The plane detection might have failed.")
        print("Try adjusting the distance_threshold parameter.")
    
    return no_floor_pcd

def main():
    parser = argparse.ArgumentParser(description="Convert point cloud to voxel grid")
    parser.add_argument('--input', type=str, required=True, help='Input PLY point cloud file')
    parser.add_argument('--output', type=str, help='Output path for voxel grid data (.npy)')
    parser.add_argument('--vis-output', type=str, help='Output directory for visualization files')
    parser.add_argument('--voxel-size', type=float, default=0.02, 
                       help='Size of voxels (default: 0.05)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize point cloud to unit cube')
    parser.add_argument('--crop-factor', type=float, default=1,
                       help='Factor to crop central region (0-1, default: 0.8 = central 80%%)')
    parser.add_argument('--clean', action='store_true',
                       help='Remove outlier points using statistical analysis')
    parser.add_argument('--nb-neighbors', type=int, default=10,
                       help='Number of neighbors to consider for outlier removal')
    parser.add_argument('--std-ratio', type=float, default=1e-4,
                       help='Standard deviation ratio for outlier removal')
    parser.add_argument('--isolate', action='store_true',
                       help='Isolate the main object using clustering')
    parser.add_argument('--cluster-eps', type=float, default=0.1,
                       help='Maximum distance between points in the same cluster')
    parser.add_argument('--min-cluster-points', type=int, default=100,
                       help='Minimum number of points to form a cluster')
    parser.add_argument('--remove-floor', action='store_true',
                       help='Remove the floor plane from the point cloud')
    parser.add_argument('--plane-threshold', type=float, default=0.1,
                       help='Maximum distance from plane for points to be considered part of it')
    args = parser.parse_args()

    try:
        # Load point cloud
        pcd = load_point_cloud(args.input)
        
        # Remove outliers if requested
        if args.clean:
            pcd = remove_outliers(pcd, nb_neighbors=args.nb_neighbors, 
                                std_ratio=args.std_ratio)
        
        # Crop to center if requested
        if args.crop_factor < 1.0:
            print(f"\nCropping to central {args.crop_factor * 100}% of the scene...")
            pcd = crop_to_center(pcd, args.crop_factor)
        
        # Always detect floor for orientation
        plane_model, floor_inliers, no_floor_pcd = detect_floor_plane(
            pcd, distance_threshold=args.plane_threshold)
        
        # Orient using floor plane
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        oriented_points, rotation = orient_by_floor(pcd, plane_model)
        
        # Apply orientation to both point clouds
        pcd.points = o3d.utility.Vector3dVector(oriented_points)
        
        # Orient the no-floor point cloud as well
        if args.remove_floor:
            no_floor_points = np.asarray(no_floor_pcd.points)
            no_floor_colors = np.asarray(no_floor_pcd.colors)
            oriented_no_floor_points, _ = orient_by_floor(no_floor_pcd, plane_model)
            no_floor_pcd.points = o3d.utility.Vector3dVector(oriented_no_floor_points)
        
        # Remove floor points if requested
        if args.remove_floor:
            pcd = no_floor_pcd
        
        # Isolate main object if requested
        if args.isolate:
            pcd = isolate_main_object(pcd, eps=args.cluster_eps,
                                    min_points=args.min_cluster_points)
        
        # Normalize if requested
        if args.normalize:
            print("\nNormalizing point cloud...")
            pcd = normalize_point_cloud(pcd)
        
        # Create voxel grid
        voxel_grid, voxel_coords, voxel_colors, grid = create_voxel_grid(pcd, args.voxel_size)
        
        # Save voxel grid if output path provided
        if args.output:
            save_voxel_grid(grid, args.output)
        
        # Save visualization files if path provided
        if args.vis_output:
            save_visualization(pcd, voxel_coords, voxel_colors, args.vis_output)
        
    except Exception as e:
        print(f"Error during voxelization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 