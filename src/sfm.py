import os
import sys
import argparse
from pathlib import Path
import subprocess
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import open3d as o3d

# Register HEIF/HEIC opener with Pillow
register_heif_opener()

def normalize_points(points):
    """Center and normalize points to unit cube"""
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit cube, but use robust statistics
    # Use percentiles instead of absolute max to avoid outlier influence
    scale = np.percentile(np.abs(points), 98)  # Use 98th percentile for scale
    points = points / scale
    
    return points

def orient_point_cloud(points):
    """Orient the point cloud using PCA so that:
    - Largest variation is along Y (height)
    - Second largest along X (width)
    - Smallest along Z (depth)
    Also ensures the object is upright based on point distribution"""
    
    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Compute principal components
    cov = np.cov(centered.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Create rotation matrix:
    # - Second component (width) becomes X
    # - First component (height) becomes Y
    # - Third component (depth) becomes Z
    R = np.zeros((3, 3))
    R[:, 0] = eigenvecs[:, 1]  # Second largest becomes X
    R[:, 1] = eigenvecs[:, 0]  # Largest becomes Y
    R[:, 2] = eigenvecs[:, 2]  # Smallest becomes Z
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    
    # Apply rotation
    aligned = np.dot(centered, R)
    
    # Make sure the object is upright
    # Check if more points are in the lower half
    if np.sum(aligned[:, 1] < 0) > len(aligned) / 2:
        # If more points are below the center, flip Y and Z
        R[:, 1] = -R[:, 1]
        R[:, 2] = -R[:, 2]
        aligned = np.dot(centered, R)
    
    return aligned + centroid, R

def run_command(cmd, description):
    """
    Run a command and print its output in real-time
    """
    print(f"\n=== {description} ===")
    print(f"Running command: {cmd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")  # Explicitly set CUDA device
    )
    
    # Print output in real-time, handling potential binary output
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            try:
                print(output.decode('utf-8'), end='')
            except UnicodeDecodeError:
                # Skip binary output that can't be decoded
                continue
    
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code {process.returncode}")

def setup_colmap_workspace(input_dir, output_dir):
    """
    Setup the COLMAP workspace with the necessary directory structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'dense'), exist_ok=True)
    
    # Verify input directory exists and contains images
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    image_files = list(Path(input_dir).glob('*.jpg')) + \
                 list(Path(input_dir).glob('*.jpeg')) + \
                 list(Path(input_dir).glob('*.png')) + \
                 list(Path(input_dir).glob('*.HEIC')) + \
                 list(Path(input_dir).glob('*.heic'))
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_files)} images in {input_dir}")
    return image_files

def convert_heic_to_jpg(image_files):
    """
    Convert HEIC images to JPG format.
    Returns a list of paths to all images (both converted and original).
    """
    final_image_files = []
    for img_path in image_files:
        if img_path.suffix.lower() in ['.heic']:
            # Create new jpg path
            jpg_path = img_path.with_suffix('.jpg')
            print(f"Converting {img_path} to {jpg_path}")
            
            # Convert HEIC to JPG
            with Image.open(img_path) as img:
                # Convert to RGB in case the HEIC is in a different color space
                img = img.convert('RGB')
                img.save(jpg_path, 'JPEG', quality=95)
            
            # Add the new jpg path to the list
            final_image_files.append(jpg_path)
            
            # Optionally remove the original HEIC file
            os.remove(img_path)
        else:
            final_image_files.append(img_path)
    
    return final_image_files

def crop_to_center(points, colors, crop_factor=0.8):
    """
    Crop the point cloud to keep only the central region around the mean point position.
    Returns cropped points and colors.
    """
    # Compute mean position (true center of mass)
    center = np.mean(points, axis=0)
    
    # Calculate extent
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    
    # Calculate new bounds based on crop factor and centered on mean position
    half_size = extent * (crop_factor / 2)
    min_bound = center - half_size
    max_bound = center + half_size
    
    # Find points within bounds
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    cropped_points = points[mask]
    cropped_colors = colors[mask]
    
    print(f"Original points: {len(points)}")
    print(f"Points after cropping: {len(cropped_points)}")
    print(f"Removed {len(points) - len(cropped_points)} points ({(1 - len(cropped_points)/len(points))*100:.1f}%)")
    print(f"Crop box center: {center}")
    print(f"Crop box size: {extent * crop_factor}")
    
    return cropped_points, cropped_colors

def run_sfm_pipeline(input_dir, output_dir, orient_cloud=False, crop_factor=0.8):
    """
    Run the complete COLMAP SfM pipeline:
    1. Feature extraction
    2. Feature matching
    3. Sparse reconstruction
    4. Dense reconstruction
    5. (Optional) Crop and orient point cloud
    """
    print("Setting up workspace...")
    image_files = setup_colmap_workspace(input_dir, output_dir)
    
    # Convert HEIC images to JPG instead of resizing
    print("Converting HEIC images to JPG format...")
    image_files = convert_heic_to_jpg([Path(f) for f in image_files])
    
    database_path = os.path.join(output_dir, 'database.db')

    # Feature extraction
    run_command(
        f"colmap feature_extractor \
        --database_path {database_path} \
        --image_path {input_dir} \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --SiftExtraction.use_gpu 1",
        "Extracting features"
    )

    # Feature matching
    run_command(
        f"colmap exhaustive_matcher \
        --database_path {database_path} \
        --SiftMatching.use_gpu 1",
        "Matching features"
    )

    # Sparse reconstruction
    sparse_dir = os.path.join(output_dir, 'sparse')
    run_command(
        f"colmap mapper \
        --database_path {database_path} \
        --image_path {input_dir} \
        --output_path {sparse_dir} \
        --Mapper.ba_refine_focal_length 1 \
        --Mapper.ba_refine_extra_params 1 \
        --Mapper.min_num_matches 15\
        --Mapper.init_min_num_inliers 100 \
        --Mapper.abs_pose_min_num_inliers 30 \
        --Mapper.abs_pose_min_inlier_ratio 0.25 \
        --Mapper.ba_local_max_num_iterations 25 \
        --Mapper.ba_global_max_num_iterations 50 \
        --Mapper.tri_min_angle 1.5",
        "Running sparse reconstruction"
    )

    # Dense reconstruction
    dense_dir = os.path.join(output_dir, 'dense')
    
    # Undistort images
    run_command(
        f"colmap image_undistorter \
        --image_path {input_dir} \
        --input_path {os.path.join(sparse_dir, '0')} \
        --output_path {dense_dir} \
        --output_type COLMAP \
        --max_image_size 1000",
        "Undistorting images"
    )

    # Compute stereo
    run_command(
        f"colmap patch_match_stereo \
        --workspace_path {dense_dir} \
        --workspace_format COLMAP \
        --PatchMatchStereo.depth_min 0.1 \
        --PatchMatchStereo.depth_max 100 \
        --PatchMatchStereo.window_radius 5 \
        --PatchMatchStereo.window_step 2 \
        --PatchMatchStereo.num_samples 15 \
        --PatchMatchStereo.num_iterations 10 \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.filter true \
        --PatchMatchStereo.filter_min_ncc 0.1 \
        --PatchMatchStereo.filter_min_triangulation_angle 2.0 \
        --PatchMatchStereo.filter_min_num_consistent 2 \
        --PatchMatchStereo.filter_geom_consistency_max_cost 1.0 \
        --PatchMatchStereo.gpu_index 0",
        "Computing stereo depth maps"
    )

    # Fuse depth maps
    point_cloud_path = os.path.join(dense_dir, 'fused.ply')
    run_command(
        f"colmap stereo_fusion \
        --workspace_path {dense_dir} \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path {point_cloud_path} \
        --StereoFusion.min_num_pixels 2 \
        --StereoFusion.max_reproj_error 2.0 \
        --StereoFusion.max_depth_error 0.1 \
        --StereoFusion.max_normal_error 10",
        "Fusing depth maps into point cloud"
    )

    # Orient point cloud if requested
    if orient_cloud:
        print("Processing point cloud...")
        oriented_path = os.path.join(dense_dir, 'fused_oriented.ply')
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # First crop to center
        print("\nCropping to central region...")
        points, colors = crop_to_center(points, colors, crop_factor)
        
        # Then orient using PCA
        print("\nOrienting using PCA...")
        oriented_points, rotation = orient_point_cloud(points)
        
        # Finally normalize
        print("\nNormalizing to unit cube...")
        normalized_points = normalize_points(oriented_points)
        
        # Save oriented point cloud
        oriented_pcd = o3d.geometry.PointCloud()
        oriented_pcd.points = o3d.utility.Vector3dVector(normalized_points)
        oriented_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(oriented_path, oriented_pcd)
        print(f"Saved oriented point cloud to {oriented_path}")
        return oriented_path

    return point_cloud_path

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP SfM pipeline on input images")
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for reconstruction')
    parser.add_argument('--orient', action='store_true', help='Orient the point cloud after reconstruction')
    parser.add_argument('--crop-factor', type=float, default=0.8,
                       help='Factor to crop central region (0-1, default: 0.8 = central 80%%)')
    args = parser.parse_args()

    try:
        point_cloud_path = run_sfm_pipeline(args.input, args.output, 
                                          orient_cloud=args.orient,
                                          crop_factor=args.crop_factor)
        print(f"\nSuccess! You can now use the point cloud at {point_cloud_path} for voxelization.")
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()