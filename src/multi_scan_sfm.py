import os
import sys
import argparse
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from typing import List, Tuple
import random
from tqdm import tqdm

# Import functions from the existing SfM script
from sfm import (
    run_sfm_pipeline, 
    normalize_points, 
    orient_point_cloud,
    crop_to_center,
    setup_colmap_workspace
)

def divide_images_into_scans(image_files: List[Path], num_scans: int, overlap_ratio: float = 0.3) -> List[List[Path]]:
    """
    Divide images into overlapping groups for multi-scan reconstruction.
    
    Args:
        image_files: List of image file paths
        num_scans: Number of scans to create
        overlap_ratio: Ratio of overlap between consecutive scans (0.0 to 1.0)
    
    Returns:
        List of image file lists, one for each scan
    """
    if num_scans <= 1:
        return [image_files]
    
    total_images = len(image_files)
    if total_images < num_scans:
        print(f"Warning: Only {total_images} images available, reducing to {total_images} scans")
        num_scans = total_images
    
    # Calculate images per scan with overlap
    base_images_per_scan = total_images // num_scans
    overlap_images = int(base_images_per_scan * overlap_ratio)
    
    scans = []
    for i in range(num_scans):
        start_idx = i * (base_images_per_scan - overlap_images)
        
        if i == num_scans - 1:  # Last scan gets remaining images
            end_idx = total_images
        else:
            end_idx = start_idx + base_images_per_scan
        
        # Ensure we don't go out of bounds
        start_idx = min(start_idx, total_images - 1)
        end_idx = min(end_idx, total_images)
        
        if start_idx < end_idx:
            scan_images = image_files[start_idx:end_idx]
            scans.append(scan_images)
            print(f"Scan {i+1}: {len(scan_images)} images (indices {start_idx}-{end_idx-1})")
    
    return scans

def rough_alignment_feature_based(source_pcd: o3d.geometry.PointCloud, 
                                target_pcd: o3d.geometry.PointCloud,
                                voxel_size: float = 0.05) -> np.ndarray:
    """
    Perform rough alignment using FPFH features and RANSAC.
    
    Args:
        source_pcd: Source point cloud to align
        target_pcd: Target point cloud to align to
        voxel_size: Voxel size for downsampling and feature computation
    
    Returns:
        4x4 transformation matrix
    """
    print("Computing rough alignment using FPFH features...")
    
    # Downsample point clouds
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    print(f"Rough alignment fitness: {result.fitness:.4f}")
    print(f"Rough alignment RMSE: {result.inlier_rmse:.6f}")
    
    return result.transformation

def icp_refinement(source_pcd: o3d.geometry.PointCloud,
                  target_pcd: o3d.geometry.PointCloud,
                  initial_transform: np.ndarray,
                  voxel_size: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Refine alignment using Iterative Closest Point (ICP).
    
    Args:
        source_pcd: Source point cloud to align
        target_pcd: Target point cloud to align to
        initial_transform: Initial transformation from rough alignment
        voxel_size: Voxel size for ICP
    
    Returns:
        Tuple of (refined_transformation_matrix, fitness_score)
    """
    print("Refining alignment using ICP...")
    
    # Apply initial transformation
    source_temp = o3d.geometry.PointCloud()
    source_temp.points = o3d.utility.Vector3dVector(np.asarray(source_pcd.points))
    source_temp.colors = o3d.utility.Vector3dVector(np.asarray(source_pcd.colors))
    if source_pcd.has_normals():
        source_temp.normals = o3d.utility.Vector3dVector(np.asarray(source_pcd.normals))
    source_temp.transform(initial_transform)
    
    # Downsample for ICP
    source_down = source_temp.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for point-to-plane ICP
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Point-to-plane ICP
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    # Combine transformations
    final_transform = np.dot(result.transformation, initial_transform)
    
    print(f"ICP refinement fitness: {result.fitness:.4f}")
    print(f"ICP refinement RMSE: {result.inlier_rmse:.6f}")
    
    return final_transform, result.fitness

def align_point_clouds(point_clouds: List[o3d.geometry.PointCloud], 
                      voxel_size: float = 0.05,
                      fitness_threshold: float = 0.3) -> List[o3d.geometry.PointCloud]:
    """
    Align multiple point clouds using pairwise registration.
    
    The first point cloud serves as the reference. Each subsequent cloud
    is aligned to the reference coordinate system.
    
    Args:
        point_clouds: List of point clouds to align
        voxel_size: Voxel size for alignment algorithms
        fitness_threshold: Minimum fitness score to accept alignment
    
    Returns:
        List of aligned point clouds
    """
    if len(point_clouds) <= 1:
        return point_clouds
    
    print(f"\nAligning {len(point_clouds)} point clouds...")
    aligned_clouds = [point_clouds[0]]  # Reference cloud
    
    for i, source_cloud in enumerate(point_clouds[1:], 1):
        print(f"\nAligning cloud {i+1} to reference...")
        
        # Try to align to the reference cloud
        target_cloud = aligned_clouds[0]
        
        try:
            # Step 1: Rough alignment using FPFH features
            rough_transform = rough_alignment_feature_based(
                source_cloud, target_cloud, voxel_size)
            
            # Step 2: ICP refinement
            final_transform, fitness = icp_refinement(
                source_cloud, target_cloud, rough_transform, voxel_size)
            
            if fitness >= fitness_threshold:
                # Apply final transformation
                aligned_cloud = o3d.geometry.PointCloud()
                aligned_cloud.points = o3d.utility.Vector3dVector(np.asarray(source_cloud.points))
                aligned_cloud.colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors))
                if source_cloud.has_normals():
                    aligned_cloud.normals = o3d.utility.Vector3dVector(np.asarray(source_cloud.normals))
                aligned_cloud.transform(final_transform)
                aligned_clouds.append(aligned_cloud)
                print(f"✓ Successfully aligned cloud {i+1} (fitness: {fitness:.4f})")
            else:
                print(f"✗ Failed to align cloud {i+1} (fitness: {fitness:.4f} < {fitness_threshold})")
                print("  Skipping this point cloud...")
        
        except Exception as e:
            print(f"✗ Error aligning cloud {i+1}: {str(e)}")
            print("  Skipping this point cloud...")
    
    print(f"\nSuccessfully aligned {len(aligned_clouds)} out of {len(point_clouds)} point clouds")
    return aligned_clouds

def merge_point_clouds(aligned_clouds: List[o3d.geometry.PointCloud],
                      voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    Merge aligned point clouds into a single point cloud.
    
    Args:
        aligned_clouds: List of aligned point clouds
        voxel_size: Voxel size for final downsampling
    
    Returns:
        Merged point cloud
    """
    print(f"\nMerging {len(aligned_clouds)} aligned point clouds...")
    
    if len(aligned_clouds) == 1:
        return aligned_clouds[0]
    
    # Combine all points and colors
    all_points = []
    all_colors = []
    
    for i, cloud in enumerate(aligned_clouds):
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)
        
        all_points.append(points)
        all_colors.append(colors)
        print(f"Cloud {i+1}: {len(points)} points")
    
    # Concatenate all points and colors
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    
    print(f"Total points before merging: {len(merged_points)}")
    
    # Create merged point cloud
    merged_cloud = o3d.geometry.PointCloud()
    merged_cloud.points = o3d.utility.Vector3dVector(merged_points)
    merged_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    
    # Remove duplicates and downsample
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size)
    print(f"Points after downsampling: {len(merged_cloud.points)}")
    
    return merged_cloud

def run_multi_scan_sfm(input_dir: str, 
                      output_dir: str,
                      num_scans: int = 3,
                      overlap_ratio: float = 0.3,
                      voxel_size: float = 0.05,
                      fitness_threshold: float = 0.3,
                      ) -> str:
    """
    Run multi-scan SfM pipeline.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        num_scans: Number of scans to create
        overlap_ratio: Overlap ratio between scans
        voxel_size: Voxel size for alignment
        fitness_threshold: Minimum fitness for accepting alignments
    
    Returns:
        Path to the final merged point cloud
    """
    print("=== Multi-Scan Structure from Motion Pipeline ===")
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = setup_colmap_workspace(input_dir, str(output_path / "temp"))
    image_files = [Path(f) for f in image_files]
    
    print(f"Found {len(image_files)} images total")
    
    # Divide images into scans
    scans = divide_images_into_scans(image_files, num_scans, overlap_ratio)
    
    if len(scans) == 1:
        print("Only one scan needed, falling back to single SfM pipeline...")
        single_output = output_path / "single_scan"
        point_cloud_path = run_sfm_pipeline(
            input_dir, str(single_output))
        
        # Copy to final output location
        final_path = str(output_path / "merged_point_cloud.ply")
        shutil.copy(point_cloud_path, final_path)
        return final_path
    
    # Run SfM for each scan
    point_clouds = []
    scan_outputs = []
    
    for i, scan_images in enumerate(scans):
        print(f"\n{'='*50}")
        print(f"Processing Scan {i+1}/{len(scans)} ({len(scan_images)} images)")
        print(f"{'='*50}")
        
        # Create temporary directory for this scan's images
        scan_dir = output_path / f"scan_{i+1}"
        scan_images_dir = scan_dir / "images"
        scan_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images for this scan
        for img_path in scan_images:
            dst_path = scan_images_dir / img_path.name
            shutil.copy(img_path, dst_path)
        
        try:
            # Run SfM pipeline for this scan
            scan_output = scan_dir / "reconstruction"
            point_cloud_path = run_sfm_pipeline(
                str(scan_images_dir), 
                str(scan_output),
            )
            
            # Load the point cloud
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            if len(pcd.points) > 0:
                point_clouds.append(pcd)
                scan_outputs.append(point_cloud_path)
                print(f"✓ Scan {i+1} successful: {len(pcd.points)} points")
            else:
                print(f"✗ Scan {i+1} failed: empty point cloud")
        
        except Exception as e:
            print(f"✗ Scan {i+1} failed: {str(e)}")
            continue
    
    if len(point_clouds) == 0:
        raise RuntimeError("No scans produced valid point clouds")
    
    print(f"\nSuccessfully reconstructed {len(point_clouds)} out of {len(scans)} scans")
    
    # Save individual point clouds before alignment
    for i, (pcd, scan_path) in enumerate(zip(point_clouds, scan_outputs)):
        individual_path = output_path / f"scan_{i+1}_cloud.ply"
        o3d.io.write_point_cloud(str(individual_path), pcd)
    
    # Align point clouds if we have more than one
    if len(point_clouds) > 1:
        aligned_clouds = align_point_clouds(
            point_clouds, voxel_size, fitness_threshold)
        
        # Save aligned point clouds
        for i, pcd in enumerate(aligned_clouds):
            aligned_path = output_path / f"scan_{i+1}_aligned.ply"
            o3d.io.write_point_cloud(str(aligned_path), pcd)
    else:
        aligned_clouds = point_clouds
    
    # Merge aligned point clouds
    merged_cloud = merge_point_clouds(aligned_clouds, voxel_size * 0.5)
    
    # Save final merged point cloud
    final_path = str(output_path / "merged_point_cloud.ply")
    o3d.io.write_point_cloud(final_path, merged_cloud)
    
    print(f"\n✓ Multi-scan reconstruction complete!")
    print(f"Final point cloud: {len(merged_cloud.points)} points")
    print(f"Saved to: {final_path}")
    
    # Cleanup temporary directories
    temp_dir = output_path / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return final_path

def main():
    parser = argparse.ArgumentParser(description="Multi-scan SfM reconstruction")
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output directory for reconstruction')
    parser.add_argument('--num-scans', type=int, default=3,
                       help='Number of scans to create (default: 3)')
    parser.add_argument('--overlap', type=float, default=0.3,
                       help='Overlap ratio between scans (default: 0.3)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size for alignment (default: 0.05)')
    parser.add_argument('--fitness-threshold', type=float, default=0.3,
                       help='Minimum fitness for accepting alignments (default: 0.3)')
    parser.add_argument('--crop-factor', type=float, default=0.8,
                       help='Factor to crop central region (default: 0.8)')
    
    args = parser.parse_args()
    
    try:
        final_path = run_multi_scan_sfm(
            args.input, 
            args.output,
            num_scans=args.num_scans,
            overlap_ratio=args.overlap,
            voxel_size=args.voxel_size,
            fitness_threshold=args.fitness_threshold,
        )
        print(f"\nSuccess! Final point cloud saved to: {final_path}")
        
    except Exception as e:
        print(f"Error during multi-scan reconstruction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()