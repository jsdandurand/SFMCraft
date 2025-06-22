import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

def smart_voxel_downsample(pcd, target_voxel_size):
    """
    Downsample point cloud using the same voxel size as the target voxel grid.
    This removes redundant points early and speeds up all subsequent processing.
    
    Args:
        pcd: Open3D point cloud
        target_voxel_size: Target voxel size for the final voxel grid
    
    Returns:
        Downsampled point cloud with one representative point per voxel
    """
    print(f"\nSmart voxel downsampling (voxel_size={target_voxel_size})...")
    original_count = len(pcd.points)
    
    # Use the target voxel size for downsampling
    # This ensures we only keep points that will contribute to different voxels
    downsampled_pcd = pcd.voxel_down_sample(target_voxel_size)
    
    final_count = len(downsampled_pcd.points)
    removed_count = original_count - final_count
    
    print(f"  Original points: {original_count}")
    print(f"  Downsampled points: {final_count}")
    print(f"  Removed {removed_count} redundant points ({removed_count/original_count*100:.1f}%)")
    print(f"  Speed improvement: ~{original_count/final_count:.1f}x faster processing")
    
    return downsampled_pcd

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

def remove_outliers(pcd, nb_neighbors=10, std_ratio=1.0):
    """Remove outliers using statistical analysis with KNN.
    A point is considered an outlier if its average distance to its k nearest neighbors
    is above std_ratio times the standard deviation of the average distances."""
    print("\nRemoving outlier points...")
    cleaned_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"Removed {len(pcd.points) - len(cleaned_pcd.points)} outlier points")
    return cleaned_pcd

def bilateral_smooth(pcd, radius=0.1, sigma_spatial=0.05, sigma_normal=0.3, num_iterations=1):
    """Apply true bilateral filtering to the point cloud to reduce noise while preserving edges.
    
    This implementation considers both spatial distance and surface normal similarity,
    using Gaussian weights for both domains to achieve edge-preserving smoothing.
    
    Args:
        pcd: Open3D point cloud
        radius: Search radius for finding neighbors
        sigma_spatial: Standard deviation for spatial Gaussian weight
        sigma_normal: Standard deviation for normal similarity Gaussian weight
        num_iterations: Number of smoothing iterations
    
    Returns:
        Smoothed point cloud with preserved edges and reduced noise
    """
    print(f"\nApplying bilateral filtering (radius={radius}, σ_spatial={sigma_spatial}, σ_normal={sigma_normal}, iterations={num_iterations})...")
    
    # Ensure normals are computed
    if not pcd.has_normals():
        pcd.estimate_normals()
    
    # Create a copy
    smoothed_pcd = o3d.geometry.PointCloud()
    smoothed_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    smoothed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    smoothed_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))
    
    points = np.asarray(smoothed_pcd.points)
    normals = np.asarray(smoothed_pcd.normals)
    colors = np.asarray(smoothed_pcd.colors)
    
    # Build KD-tree for neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    new_points = points.copy()
    
    for i, (point, normal) in enumerate(zip(points, normals)):
        # Find neighbors within radius
        [_, idx, distances] = kdtree.search_radius_vector_3d(point, radius)
        
        if len(idx) > 1:  # Need at least one neighbor besides itself
            neighbor_points = points[idx]
            neighbor_normals = normals[idx]
            
            # Compute weights based on spatial distance and normal similarity
            spatial_weights = np.exp(-np.array(distances) / (2 * sigma_spatial**2))
            
            # Compute normal similarity (dot product)
            normal_similarities = np.dot(neighbor_normals, normal)
            normal_weights = np.exp(-(1 - normal_similarities) / (2 * sigma_normal**2))
            
            # Combine weights
            total_weights = spatial_weights * normal_weights
            total_weights /= np.sum(total_weights)  # Normalize
            
            # Compute weighted average position
            new_points[i] = np.sum(neighbor_points * total_weights[:, np.newaxis], axis=0)
    
    # Update point cloud
    smoothed_pcd.points = o3d.utility.Vector3dVector(new_points)
    smoothed_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Re-estimate normals after smoothing
    smoothed_pcd.estimate_normals()

    print(f"Smoothed point cloud: {len(smoothed_pcd.points)} points")
    return smoothed_pcd

def isolate_main_object(pcd, eps=0.01, min_points=100, method="largest", camera_centers=None, cluster_size_threshold=0.05, top_k_clusters=1):
    """Isolate the main object using DBSCAN clustering with different selection strategies.
    
    Args:
        pcd: Open3D point cloud
        eps: maximum distance between points in the same cluster
        min_points: minimum number of points to form a cluster
        method: selection method - "largest", "density", "center_priority", or "camera_centered"
        camera_centers: numpy array of camera center positions [N, 3] (required for "camera_centered" method)
        cluster_size_threshold: minimum cluster size as fraction of largest cluster
        top_k_clusters: number of top clusters to keep (default: 1)
    """
    print(f"\nIsolating main object using DBSCAN clustering (method: {method})...")
    
    # Run DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    
    if len(np.unique(labels)) <= 1:
        print("Warning: No clusters found. Try adjusting eps and min_points parameters.")
        return pcd
    
    # Remove noise points (labeled as -1)
    n_noise = len(labels[labels == -1])
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
    
    print(f"Found {len(unique_labels)} clusters")
    print(f"Removed {n_noise} noise points")
    
    if len(unique_labels) == 0:
        print("Warning: All points were classified as noise")
        return pcd
    
    # Get points array for analysis
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Calculate minimum size threshold as percentage of largest cluster
    largest_cluster_size = max(np.sum(labels == label) for label in unique_labels)
    min_size_threshold = max(min_points, int(largest_cluster_size * cluster_size_threshold))
    print(f"Largest cluster has {largest_cluster_size} points")
    print(f"Minimum cluster size threshold: {min_size_threshold} points ({cluster_size_threshold*100:.1f}% of largest)")
    
    # Score all clusters based on method and select top k
    cluster_scores = []
    
    if method == "largest":
        print(f"\nScoring clusters by size:")
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size >= min_size_threshold:
                score = cluster_size  # Higher is better
                cluster_scores.append((label, score, cluster_size))
                print(f"  Cluster {label}: {cluster_size} points, score={score}")
    
    elif method == "density":
        print(f"\nScoring clusters by density:")
        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask]
            cluster_size = len(cluster_points)
            
            if cluster_size >= min_size_threshold:
                # Calculate bounding box volume
                cluster_bbox = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                cluster_volume = np.prod(cluster_bbox + 1e-8)  # Add epsilon to avoid division by zero
                score = cluster_size / cluster_volume  # Higher is better
                cluster_scores.append((label, score, cluster_size))
                print(f"  Cluster {label}: {cluster_size} points, volume={cluster_volume:.6f}, score={score:.1f}")
    
    elif method == "center_priority":
        print(f"\nScoring clusters by distance to scene center:")
        scene_center = np.mean(points, axis=0)
        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask]
            cluster_size = len(cluster_points)
            
            if cluster_size >= min_size_threshold:
                cluster_center = np.mean(cluster_points, axis=0)
                distance = np.linalg.norm(cluster_center - scene_center)
                score = -distance  # Lower distance is better, so negate for sorting
                cluster_scores.append((label, score, cluster_size))
                print(f"  Cluster {label}: {cluster_size} points, distance={distance:.3f}, score={score:.3f}")
    
    elif method == "camera_centered":
        if camera_centers is None:
            raise ValueError("camera_centers must be provided for 'camera_centered' method")
        
        print(f"\nScoring clusters by distance to camera centroid:")
        camera_centroid = np.mean(camera_centers, axis=0)
        print(f"Camera centroid: {camera_centroid}")
        for label in unique_labels:
            mask = labels == label
            cluster_points = points[mask]
            cluster_size = len(cluster_points)
            
            if cluster_size >= min_size_threshold:
                cluster_center = np.mean(cluster_points, axis=0)
                distance = np.linalg.norm(cluster_center - camera_centroid)
                score = -distance  # Lower distance is better, so negate for sorting
                cluster_scores.append((label, score, cluster_size))
                print(f"  Cluster {label}: {cluster_size} points, distance={distance:.3f}, score={score:.3f}")
    
    else:
        raise ValueError(f"Unknown isolation method: {method}")
    
    if not cluster_scores:
        print("Warning: No clusters met the size threshold")
        return pcd
    
    # Sort by score (highest first) and select top k
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = min(top_k_clusters, len(cluster_scores))
    selected_clusters = cluster_scores[:top_k]
    
    print(f"\nSelected top {top_k} clusters:")
    all_selected_points = []
    all_selected_colors = []
    total_points = 0
    
    for i, (label, score, cluster_size) in enumerate(selected_clusters, 1):
        mask = labels == label
        cluster_points = points[mask]
        cluster_colors = colors[mask]
        
        all_selected_points.append(cluster_points)
        all_selected_colors.append(cluster_colors)
        total_points += len(cluster_points)
        
        print(f"  {i}. Cluster {label}: {cluster_size} points, score={score:.3f}")
    
    # Combine all selected clusters
    if all_selected_points:
        combined_points = np.vstack(all_selected_points)
        combined_colors = np.vstack(all_selected_colors)
        
        # Create new point cloud with selected clusters
        isolated_pcd = o3d.geometry.PointCloud()
        isolated_pcd.points = o3d.utility.Vector3dVector(combined_points)
        isolated_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        print(f"Kept {top_k} clusters with {total_points} total points")
        print(f"Removed {len(pcd.points) - total_points} points from other clusters")
        
        return isolated_pcd
    else:
        return pcd

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

def postprocess_point_cloud(input_path, output_path, 
                           smart_downsample=False, downsample_factor=0.5, voxel_size=0.05,
                           crop_factor=1.0, clean=False, nb_neighbors=10, std_ratio=1e-4,
                           smooth=False, smooth_radius=0.1, sigma_spatial=0.05, sigma_normal=0.3, smooth_iterations=1,
                           isolate=False, cluster_eps=0.1, min_cluster_points=100, isolation_method="largest",
                           remove_floor_flag=False, plane_threshold=0.1,
                           orient_by_floor_flag=False, normalize=False,
                           crop_to_center_flag=False, center_crop_factor=0.8,
                           colmap_sparse_dir=None, cluster_size_threshold=0.05, top_k_clusters=1):
    """
    Complete point cloud post-processing pipeline.
    """
    print("=== Point Cloud Post-Processing Pipeline ===")
    
    # Load point cloud
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading point cloud from {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in {input_path}")
    
    print(f"Loaded {len(pcd.points)} points")
    
    # Load camera poses early if needed for camera-based isolation
    # We'll apply the same transformations to them as we do to the point cloud
    camera_centers = None
    if isolate and isolation_method in ["camera_centered"]:
        if colmap_sparse_dir is None:
            print("Warning: camera-based method requires colmap_sparse_dir. Falling back to largest method.")
            isolation_method = "largest"
        else:
            try:
                camera_centers = read_colmap_cameras_and_images(colmap_sparse_dir)
                print(f"Loaded camera poses for synchronized transformation")
            except Exception as e:
                print(f"Warning: Failed to read camera poses: {e}. Falling back to largest method.")
                isolation_method = "largest"
    
    # Smart downsampling FIRST to speed up all subsequent operations
    if smart_downsample:
        downsample_voxel_size = voxel_size * downsample_factor
        pcd = smart_voxel_downsample(pcd, downsample_voxel_size)
    
        # Remove outliers if requested (now much faster due to fewer points)
    if clean:
        pcd = remove_outliers(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    
    # Crop to center if requested
    if crop_factor < 1.0:
        print(f"\nCropping to central {crop_factor * 100}% of the scene...")
        pcd = crop_to_center(pcd, crop_factor)
    
    # Crop to center if requested (new flag)
    if crop_to_center_flag:
        print(f"\nCropping to central {center_crop_factor * 100}% of the scene (center crop)...")
        pcd = crop_to_center(pcd, center_crop_factor)
    
    # Always detect floor for orientation if needed
    plane_model = None
    if orient_by_floor_flag or remove_floor_flag:
        plane_model, floor_inliers, no_floor_pcd = detect_floor_plane(
            pcd, distance_threshold=plane_threshold)
        
        # Orient using floor plane if requested
        if orient_by_floor_flag:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            oriented_points, rotation = orient_by_floor(pcd, plane_model)
            
            # Apply orientation to point cloud
            pcd.points = o3d.utility.Vector3dVector(oriented_points)
            
            # Apply the same transformation to camera poses to keep them synchronized
            if camera_centers is not None:
                # Center camera poses the same way as point cloud
                points_centroid = np.mean(points, axis=0)
                centered_cameras = camera_centers - points_centroid
                # Apply the same rotation
                oriented_cameras = np.dot(centered_cameras, rotation)
                # Add back the centroid
                camera_centers = oriented_cameras + points_centroid
                print(f"Applied floor orientation to camera poses")
            
            # Orient the no-floor point cloud as well if we'll use it
            if remove_floor_flag:
                no_floor_points = np.asarray(no_floor_pcd.points)
                oriented_no_floor_points, _ = orient_by_floor(no_floor_pcd, plane_model)
                no_floor_pcd.points = o3d.utility.Vector3dVector(oriented_no_floor_points)
        
        # Remove floor points if requested
        if remove_floor_flag:
            pcd = no_floor_pcd
    
    # Isolate main object if requested
    if isolate:
        # Camera poses were already loaded and transformed earlier (if needed)
        
        # Adapt clustering parameters based on whether we downsampled
        if smart_downsample:
            adaptive_eps = voxel_size * downsample_factor * 2
            adaptive_min_points = max(5, min_cluster_points // 10)
            print(f"Using adaptive clustering parameters: eps={adaptive_eps:.4f}, min_points={adaptive_min_points}")
            pcd = isolate_main_object(pcd, eps=adaptive_eps, min_points=adaptive_min_points, 
                                    method=isolation_method, camera_centers=camera_centers,
                                    cluster_size_threshold=cluster_size_threshold, top_k_clusters=top_k_clusters)
        else:
            pcd = isolate_main_object(pcd, eps=cluster_eps, min_points=min_cluster_points, 
                                    method=isolation_method, camera_centers=camera_centers,
                                    cluster_size_threshold=cluster_size_threshold, top_k_clusters=top_k_clusters)

    # Bilateral smoothing if requested
    if smooth:
        pcd = bilateral_smooth(pcd, radius=smooth_radius, sigma_spatial=sigma_spatial, 
                              sigma_normal=sigma_normal, num_iterations=smooth_iterations)
    # Normalize if requested
    if normalize:
        print("\nNormalizing point cloud...")
        # Get transformation parameters before normalization
        if camera_centers is not None:
            bbox = pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            scale = np.max(bbox.get_extent())
        
        pcd = normalize_point_cloud(pcd)
        
        # Apply same normalization to camera poses
        if camera_centers is not None:
            camera_centers = (camera_centers - center) * (1.0 / scale)
            print("Applied normalization to camera poses")
    
    # Save processed point cloud
    print(f"\nSaving processed point cloud to {output_path}")
    o3d.io.write_point_cloud(output_path, pcd)
    
    print(f"\n✓ Post-processing complete!")
    print(f"Final point cloud: {len(pcd.points)} points")
    print(f"Saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Post-process point clouds for better voxelization")
    parser.add_argument('--input', type=str, required=True, help='Input PLY point cloud file')
    parser.add_argument('--output', type=str, required=True, help='Output PLY point cloud file')
    
    # Smart downsampling parameters
    parser.add_argument('--smart-downsample', action='store_true',
                       help='Downsample using target voxel size to speed up processing (recommended)')
    parser.add_argument('--downsample-factor', type=float, default=0.5,
                       help='Downsample factor relative to voxel size (default: 0.5)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Target voxel size for reference (default: 0.05)')
    
    # Cropping parameters
    parser.add_argument('--crop-factor', type=float, default=1.0,
                       help='Factor to crop central region (0-1, default: 1.0 = no cropping)')
    parser.add_argument('--crop-to-center', action='store_true',
                       help='Crop to the central region of the point cloud (center crop)')
    parser.add_argument('--center-crop-factor', type=float, default=0.8,
                       help='Factor for center crop (0-1, default: 0.8 = central 80%%)')
    
    # Cleaning parameters
    parser.add_argument('--clean', action='store_true',
                       help='Remove outlier points using statistical analysis')
    parser.add_argument('--nb-neighbors', type=int, default=10,
                       help='Number of neighbors for outlier removal (default: 10)')
    parser.add_argument('--std-ratio', type=float, default=1e-4,
                       help='Standard deviation ratio for outlier removal (default: 1e-4)')
    
    # Smoothing parameters
    parser.add_argument('--smooth', action='store_true',
                       help='Apply bilateral filtering to reduce noise while preserving edges')
    parser.add_argument('--smooth-radius', type=float, default=0.1,
                       help='Search radius for bilateral filtering (default: 0.1)')
    parser.add_argument('--sigma-spatial', type=float, default=0.05,
                       help='Spatial standard deviation for bilateral filtering (default: 0.05)')
    parser.add_argument('--sigma-normal', type=float, default=0.3,
                       help='Normal similarity standard deviation for bilateral filtering (default: 0.3)')
    parser.add_argument('--smooth-iterations', type=int, default=1,
                       help='Number of bilateral filtering iterations (default: 1)')
    
    # Isolation parameters
    parser.add_argument('--isolate', action='store_true',
                       help='Isolate the main object using clustering')
    parser.add_argument('--cluster-eps', type=float, default=0.1,
                       help='Maximum distance between points in the same cluster (default: 0.1)')
    parser.add_argument('--min-cluster-points', type=int, default=100,
                       help='Absolute minimum number of points to form a cluster (default: 100)')
    parser.add_argument('--cluster-size-threshold', type=float, default=0.05,
                       help='Minimum cluster size as fraction of largest cluster (default: 0.05 = 5%)')
    parser.add_argument('--top-k-clusters', type=int, default=1,
                       help='Number of top-scoring clusters to keep (default: 1)')
    parser.add_argument('--isolation-method', type=str, default='largest',
                       choices=['largest', 'density', 'center_priority', 'camera_centered'],
                       help='Method for selecting object cluster (default: largest)')
    parser.add_argument('--colmap-sparse-dir', type=str, default=None,
                       help='Path to COLMAP sparse reconstruction directory (required for camera_centered method)')
    
    # Floor processing parameters
    parser.add_argument('--remove-floor', action='store_true',
                       help='Remove the floor plane from the point cloud')
    parser.add_argument('--orient-by-floor', action='store_true',
                       help='Orient the point cloud using the detected floor plane')
    parser.add_argument('--plane-threshold', type=float, default=0.1,
                       help='Distance threshold for floor plane detection (default: 0.1)')
    
    # Normalization parameters
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize point cloud to unit cube')
    
    args = parser.parse_args()
    
    try:
        postprocess_point_cloud(
            args.input, args.output,
            smart_downsample=args.smart_downsample,
            downsample_factor=args.downsample_factor,
            voxel_size=args.voxel_size,
            crop_factor=args.crop_factor,
            clean=args.clean,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
            smooth=args.smooth,
            smooth_radius=args.smooth_radius,
            sigma_spatial=args.sigma_spatial,
            sigma_normal=args.sigma_normal,
            smooth_iterations=args.smooth_iterations,
            isolate=args.isolate,
            cluster_eps=args.cluster_eps,
            min_cluster_points=args.min_cluster_points,
            isolation_method=args.isolation_method,
            remove_floor_flag=args.remove_floor,
            orient_by_floor_flag=args.orient_by_floor,
            plane_threshold=args.plane_threshold,
            normalize=args.normalize,
            crop_to_center_flag=args.crop_to_center,
            center_crop_factor=args.center_crop_factor,
            colmap_sparse_dir=args.colmap_sparse_dir,
            cluster_size_threshold=args.cluster_size_threshold,
            top_k_clusters=args.top_k_clusters
        )
        
    except Exception as e:
        print(f"Error during post-processing: {str(e)}", file=sys.stderr)
        sys.exit(1)

def read_colmap_cameras_and_images(sparse_dir):
    """
    Read COLMAP camera poses from sparse reconstruction output.
    
    Args:
        sparse_dir: Path to COLMAP sparse reconstruction directory (usually output/sparse/0)
    
    Returns:
        camera_centers: numpy array of camera center positions [N, 3]
    """
    import struct
    
    # Try to read binary format first, then fall back to text format
    images_bin_path = os.path.join(sparse_dir, "images.bin")
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    
    camera_centers = []
    
    if os.path.exists(images_bin_path):
        # Read binary format
        print("Reading camera poses from COLMAP binary format...")
        with open(images_bin_path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                # Read image header
                image_id = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
                tx, ty, tz = struct.unpack("<ddd", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]
                
                # Read image name
                name_len = 0
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name_len += 1
                f.seek(-name_len - 1, 1)  # Go back
                name = f.read(name_len + 1)[:-1].decode('utf-8')
                
                # Read 2D points
                num_points2D = struct.unpack("<Q", f.read(8))[0]
                f.read(24 * num_points2D)  # Skip 2D points data
                
                # Convert quaternion and translation to camera center
                # COLMAP uses world-to-camera transformation
                # Camera center = -R^T * t where R is rotation matrix from quaternion
                R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
                t = np.array([tx, ty, tz])
                camera_center = -R.T @ t
                camera_centers.append(camera_center)
                
    elif os.path.exists(images_txt_path):
        # Read text format
        print("Reading camera poses from COLMAP text format...")
        with open(images_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 10:
                    # Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    
                    # Convert quaternion and translation to camera center
                    R = quaternion_to_rotation_matrix([qw, qx, qy, qz])
                    t = np.array([tx, ty, tz])
                    camera_center = -R.T @ t
                    camera_centers.append(camera_center)
    else:
        raise FileNotFoundError(f"No camera pose files found in {sparse_dir}")
    
    if not camera_centers:
        raise ValueError(f"No camera poses found in {sparse_dir}")
        
    camera_centers = np.array(camera_centers)
    print(f"Found {len(camera_centers)} camera poses")
    return camera_centers

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: quaternion as [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

if __name__ == '__main__':
    main()