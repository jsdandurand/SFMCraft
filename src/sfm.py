import os
import sys
import argparse
from pathlib import Path
import subprocess

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
    
    image_files = list(Path(input_dir).glob('*.jpg')) + list(Path(input_dir).glob('*.jpeg')) + list(Path(input_dir).glob('*.png'))
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_files)} images in {input_dir}")
    return image_files

def run_sfm_pipeline(input_dir, output_dir):
    """
    Run the complete COLMAP SfM pipeline:
    1. Feature extraction
    2. Feature matching
    3. Sparse reconstruction
    4. Dense reconstruction
    """
    print("Setting up workspace...")
    image_files = setup_colmap_workspace(input_dir, output_dir)
    database_path = os.path.join(output_dir, 'database.db')

    # Feature extraction
    run_command(
        f"colmap feature_extractor \
        --database_path {database_path} \
        --image_path {input_dir} \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --ImageReader.single_camera 1 \
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
        --Mapper.min_num_matches 15 \
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
        --PatchMatchStereo.num_samples 7 \
        --PatchMatchStereo.num_iterations 3 \
        --PatchMatchStereo.geom_consistency true \
        --PatchMatchStereo.filter true \
        --PatchMatchStereo.filter_min_ncc 0.1 \
        --PatchMatchStereo.filter_min_triangulation_angle 3.0 \
        --PatchMatchStereo.filter_min_num_consistent 2 \
        --PatchMatchStereo.filter_geom_consistency_max_cost 1.0 \
        --PatchMatchStereo.gpu_index 0",
        "Computing stereo depth maps"
    )

    # Fuse depth maps
    run_command(
        f"colmap stereo_fusion \
        --workspace_path {dense_dir} \
        --workspace_format COLMAP \
        --input_type geometric \
        --output_path {os.path.join(dense_dir, 'fused.ply')} \
        --StereoFusion.min_num_pixels 2 \
        --StereoFusion.max_reproj_error 2.0 \
        --StereoFusion.max_depth_error 0.1 \
        --StereoFusion.max_normal_error 10",
        "Fusing depth maps into point cloud"
    )

    point_cloud_path = os.path.join(dense_dir, 'fused.ply')
    return point_cloud_path

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP SfM pipeline on input images")
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for reconstruction')
    args = parser.parse_args()

    try:
        point_cloud_path = run_sfm_pipeline(args.input, args.output)
        print(f"\nSuccess! You can now use the point cloud at {point_cloud_path} for voxelization.")
    except Exception as e:
        print(f"Error during reconstruction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 