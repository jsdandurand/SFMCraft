import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import tempfile

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    if description:
        print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    print(f"✓ {description} completed successfully")

def extract_video_frames(video_path, output_dir, n_images=100, max_frames=None, **kwargs):
    """Extract frames from video using sample_video_frames.py"""
    cmd = [
        sys.executable, "src/sample_video_frames.py",
        "--video", str(video_path),
        "--output", str(output_dir),
        "--n_images", str(n_images)
    ]
    
    # Add lighting correction parameters
    if kwargs.get("enable_lighting_correction", False):
        cmd.append("--enable-lighting-correction")
    if "clahe_clip_limit" in kwargs:
        cmd.extend(["--clahe-clip-limit", str(kwargs["clahe_clip_limit"])])
    if "clahe_tile_size" in kwargs:
        cmd.extend(["--clahe-tile-size", str(kwargs["clahe_tile_size"])])
    if "gamma_correction" in kwargs:
        cmd.extend(["--gamma-correction", str(kwargs["gamma_correction"])])
    if "exposure_adjustment" in kwargs:
        cmd.extend(["--exposure-adjustment", str(kwargs["exposure_adjustment"])])
    if "brightness_adjustment" in kwargs:
        cmd.extend(["--brightness-adjustment", str(kwargs["brightness_adjustment"])])
    if "contrast_adjustment" in kwargs:
        cmd.extend(["--contrast-adjustment", str(kwargs["contrast_adjustment"])])
    if kwargs.get("enhance_texture", False):
        cmd.append("--enhance-texture")
    if "texture_enhancement_strength" in kwargs:
        cmd.extend(["--texture-enhancement-strength", str(kwargs["texture_enhancement_strength"])])
    if kwargs.get("reduce_bright_spots", False):
        cmd.append("--reduce-bright-spots")
    
    run_command(cmd, "Video frame extraction")

def run_sfm(input_dir, output_dir, method="single", **kwargs):
    """Run SfM reconstruction (single or multi-scan)"""
    if method == "multi":
        cmd = [
            sys.executable, "src/multi_scan_sfm.py",
            "--input", str(input_dir),
            "--output", str(output_dir)
        ]
        
        # Add multi-scan specific parameters
        if "num_scans" in kwargs:
            cmd.extend(["--num-scans", str(kwargs["num_scans"])])
        if "overlap" in kwargs:
            cmd.extend(["--overlap", str(kwargs["overlap"])])
        if "voxel_size" in kwargs:
            cmd.extend(["--voxel-size", str(kwargs["voxel_size"])])
        if "fitness_threshold" in kwargs:
            cmd.extend(["--fitness-threshold", str(kwargs["fitness_threshold"])])
        
        run_command(cmd, "Multi-scan SfM reconstruction")
        return str(Path(output_dir) / "merged_point_cloud.ply")
    
    else:  # single scan
        cmd = [
            sys.executable, "src/sfm.py",
            "--input", str(input_dir),
            "--output", str(output_dir)
        ]
        
        run_command(cmd, "Single SfM reconstruction")
        return str(Path(output_dir) / "dense" / "fused.ply")

def run_postprocessing(input_path, output_path, **kwargs):
    """Run point cloud postprocessing"""
    cmd = [
        sys.executable, "src/postprocess_pointcloud.py",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    
    # Add postprocessing parameters
    if kwargs.get("smart_downsample", False):
        cmd.append("--smart-downsample")
    if "downsample_factor" in kwargs:
        cmd.extend(["--downsample-factor", str(kwargs["downsample_factor"])])
    if "voxel_size" in kwargs:
        cmd.extend(["--voxel-size", str(kwargs["voxel_size"])])
    if kwargs.get("clean", False):
        cmd.append("--clean")

    if kwargs.get("smooth", False):
        cmd.append("--smooth")
    if "smooth_radius" in kwargs:
        cmd.extend(["--smooth-radius", str(kwargs["smooth_radius"])])
    if "sigma_spatial" in kwargs:
        cmd.extend(["--sigma-spatial", str(kwargs["sigma_spatial"])])
    if "sigma_normal" in kwargs:
        cmd.extend(["--sigma-normal", str(kwargs["sigma_normal"])])
    if "smooth_iterations" in kwargs:
        cmd.extend(["--smooth-iterations", str(kwargs["smooth_iterations"])])
    if kwargs.get("isolate", False):
        cmd.append("--isolate")

    if "cluster_size_threshold" in kwargs:
        cmd.extend(["--cluster-size-threshold", str(kwargs["cluster_size_threshold"])])
    if "top_k_clusters" in kwargs:
        cmd.extend(["--top-k-clusters", str(kwargs["top_k_clusters"])])
    if "isolation_method" in kwargs:
        cmd.extend(["--isolation-method", str(kwargs["isolation_method"])])
    if "colmap_sparse_dir" in kwargs and kwargs["colmap_sparse_dir"]:
        cmd.extend(["--colmap-sparse-dir", str(kwargs["colmap_sparse_dir"])])
    if kwargs.get("remove_floor", False):
        cmd.append("--remove-floor")
    if kwargs.get("orient_by_floor", False):
        cmd.append("--orient-by-floor")
    if kwargs.get("normalize", False):
        cmd.append("--normalize")
    if kwargs.get("crop_to_center", False):
        cmd.append("--crop-to-center")
    if "center_crop_factor" in kwargs:
        cmd.extend(["--center-crop-factor", str(kwargs["center_crop_factor"])])
    if kwargs.get("remove_white_flag", False):
        cmd.append("--remove-white")
    if "white_threshold" in kwargs:
        cmd.extend(["--white-threshold", str(kwargs["white_threshold"])])
    
    run_command(cmd, "Point cloud postprocessing")

def run_voxelization(input_path, output_path, **kwargs):
    """Run voxelization"""
    cmd = [
        sys.executable, "src/voxelize.py",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    
    # Add voxelization parameters
    if "voxel_size" in kwargs:
        cmd.extend(["--voxel-size", str(kwargs["voxel_size"])])
    if kwargs.get("normalize", False):
        cmd.append("--normalize")
    if "vis_output" in kwargs:
        cmd.extend(["--vis-output", str(kwargs["vis_output"])])
    
    run_command(cmd, "Voxelization")

def run_model_viewer(voxel_path):
    """Run model viewer"""
    cmd = [
        sys.executable, "src/view_model.py",
        "--voxel-grid", str(voxel_path)
    ]
    
    run_command(cmd, "Model viewer")

def main():
    parser = argparse.ArgumentParser(description="Complete Image2MC pipeline")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images', type=str, help='Directory containing input images')
    input_group.add_argument('--video', type=str, help='Input video file')
    
    # Output directory
    parser.add_argument('--output', type=str, required=True, help='Output directory for all results')
    
    # Video extraction options
    parser.add_argument('--n_images', type=int, default=100,
                       help='Number of images to sample from video (default: 100)')
    
    # Lighting correction options (for video frame extraction)
    parser.add_argument('--enable-lighting-correction', action='store_true',
                       help='Apply lighting preprocessing to video frames to reduce white voxels from overexposure')
    parser.add_argument('--enable-clahe', action='store_true',
                       help='Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting')
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0,
                       help='Clip limit for CLAHE (default: 2.0, higher = more contrast)')
    parser.add_argument('--clahe-tile-size', type=int, default=8,
                       help='Tile grid size for CLAHE (default: 8x8)')
    parser.add_argument('--gamma-correction', type=float, default=1.0,
                       help='Gamma correction value (1.0 = no change, <1.0 = brighter, >1.0 = darker)')
    parser.add_argument('--exposure-adjustment', type=int, default=0,
                       help='Exposure adjustment (-100 to +100, negative = darker, positive = brighter)')
    parser.add_argument('--brightness-adjustment', type=int, default=0,
                       help='Brightness adjustment (-100 to +100)')
    parser.add_argument('--contrast-adjustment', type=float, default=1.0,
                       help='Contrast adjustment (0.5 to 3.0, 1.0 = no change)')
    parser.add_argument('--enhance-texture', action='store_true',
                       help='Apply texture enhancement for featureless surfaces (helps with white/plain objects)')
    parser.add_argument('--texture-enhancement-strength', type=float, default=1.5,
                       help='Strength of texture enhancement (1.0-3.0, default: 1.5)')
    parser.add_argument('--reduce-bright-spots', action='store_true',
                       help='Apply preset parameters for reducing specular reflections and bright spots (auto-sets optimal lighting correction values)')
    
    # SfM method selection
    parser.add_argument('--sfm-method', type=str, choices=['single', 'multi'], default='single',
                       help='SfM reconstruction method (default: single)')
    parser.add_argument('--force-sfm', action='store_true',
                       help='Force SfM reconstruction even if output already exists')
    
    # Multi-scan SfM parameters
    parser.add_argument('--num-scans', type=int, default=3,
                       help='Number of scans for multi-scan SfM (default: 3)')
    parser.add_argument('--overlap', type=float, default=0.3,
                       help='Overlap ratio between scans (default: 0.3)')
    parser.add_argument('--fitness-threshold', type=float, default=0.3,
                       help='Minimum fitness for accepting alignments (default: 0.3)')
    
    # SfM general parameters
    # Postprocessing parameters
    
    parser.add_argument('--skip-postprocess', action='store_true',
                       help='Skip postprocessing step')
    parser.add_argument('--smart-downsample', action='store_true',
                       help='Apply smart voxel downsampling during postprocessing')
    parser.add_argument('--downsample-factor', type=float, default=0.5,
                       help='Downsample factor relative to voxel size (default: 0.5)')
    parser.add_argument('--clean', action='store_true',
                       help='Remove outlier points during postprocessing')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply bilateral filtering to reduce noise while preserving edges')
    parser.add_argument('--smooth-radius', type=float, default=0.1,
                       help='Search radius for bilateral filtering (default: 0.1)')
    parser.add_argument('--sigma-spatial', type=float, default=0.05,
                       help='Spatial standard deviation for bilateral filtering (default: 0.05)')
    parser.add_argument('--sigma-normal', type=float, default=0.5,
                       help='Normal similarity standard deviation for bilateral filtering (default: 0.3)')
    parser.add_argument('--smooth-iterations', type=int, default=5,
                       help='Number of bilateral filtering iterations (default: 1)')
    parser.add_argument('--isolate', action='store_true',
                       help='Isolate main object using clustering')
    parser.add_argument('--cluster-size-threshold', type=float, default=0.01,
                       help='Minimum cluster size as fraction of largest cluster (default: 0.01 = 1%)')
    parser.add_argument('--top-k-clusters', type=int, default=1,
                       help='Number of top-scoring clusters to keep (default: 1)')
    parser.add_argument('--isolation-method', type=str, default='camera_centered',
                       choices=['largest', 'density', 'center_priority', 'camera_centered'],
                       help='Method for selecting object cluster (default: largest)')

    parser.add_argument('--remove-floor', action='store_true',
                       help='Remove floor plane during postprocessing')
    parser.add_argument('--orient-by-floor', action='store_true',
                       help='Orient point cloud using detected floor plane')
    parser.add_argument('--normalize-postprocess', action='store_true',
                       help='Normalize point cloud during postprocessing')
    
    # Voxelization parameters
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='Voxel size for grid creation (default: 0.05)')
    parser.add_argument('--normalize-voxel', action='store_true',
                       help='Normalize point cloud before voxelization')
    parser.add_argument('--save-vis', action='store_true',
                       help='Save visualization files during voxelization')
    
    # Viewer option
    parser.add_argument('--view', action='store_true',
                       help='Launch model viewer after completion')
    
    # Cleanup option
    parser.add_argument('--keep-intermediate', action='store_true',
                       help='Keep intermediate files (default: cleanup)')
    
    # New options for center cropping
    parser.add_argument('--crop-to-center', action='store_true',
                       help='Crop to the central region of the point cloud (center crop)')
    parser.add_argument('--center-crop-factor', type=float, default=0.8,
                       help='Factor for center crop (0-1, default: 0.8 = central 80%%)')
    
    # New options for removing white points
    parser.add_argument('--remove-white', action='store_true',
                       help='Remove white or near-white points based on color intensity during postprocessing')
    parser.add_argument('--white-threshold', type=int, default=240,
                       help='RGB threshold above which points are considered white (default: 240)')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing SfM output first to avoid unnecessary video frame extraction
    sfm_output_dir = output_dir / "sfm_output"
    
    # Determine expected point cloud path based on SfM method
    if args.sfm_method == "multi":
        expected_point_cloud = sfm_output_dir / "merged_point_cloud.ply"
    else:
        expected_point_cloud = sfm_output_dir / "dense" / "fused.ply"
    
    # Also check for saved original point cloud from previous run
    saved_original_cloud = output_dir / "original_point_cloud.ply"
    
    # Determine if we need to run SfM
    need_sfm = args.force_sfm or (not expected_point_cloud.exists() and not saved_original_cloud.exists())
    
    # Determine input directory (only extract video frames if we need to run SfM)
    if args.video:
        print(f"Input: Video file {args.video}")
        
        if need_sfm:
            # Extract frames from video only if we need to run SfM
            frames_dir = output_dir / "frames"
            
            # Prepare lighting correction kwargs
            lighting_kwargs = {
                "enable_lighting_correction": args.enable_lighting_correction or args.reduce_bright_spots,
                "enable_clahe": args.enable_clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
                "clahe_tile_size": args.clahe_tile_size,
                "gamma_correction": args.gamma_correction,
                "exposure_adjustment": args.exposure_adjustment,
                "brightness_adjustment": args.brightness_adjustment,
                "contrast_adjustment": args.contrast_adjustment,
                "enhance_texture": args.enhance_texture,
                "texture_enhancement_strength": args.texture_enhancement_strength,
                "reduce_bright_spots": args.reduce_bright_spots,
            }
            
            extract_video_frames(
                args.video, 
                frames_dir, 
                n_images=args.n_images,
                **lighting_kwargs
            )
            input_images_dir = frames_dir
        else:
            # We have existing SfM output, so we don't need frames
            print("Existing SfM output found - skipping video frame extraction")
            input_images_dir = None  # We won't use this
    else:
        print(f"Input: Image directory {args.images}")
        input_images_dir = Path(args.images)
        if need_sfm and not input_images_dir.exists():
            print(f"ERROR: Input directory {input_images_dir} does not exist")
            sys.exit(1)
    
    # Run SfM reconstruction if needed
    if not need_sfm and expected_point_cloud.exists():
        print(f"\n{'='*60}")
        print("EXISTING SfM OUTPUT FOUND - SKIPPING SfM RECONSTRUCTION")
        print(f"{'='*60}")
        print(f"Using existing point cloud: {expected_point_cloud}")
        print("(Use --force-sfm to override this behavior)")
        point_cloud_path = str(expected_point_cloud)
    elif not need_sfm and saved_original_cloud.exists():
        print(f"\n{'='*60}")
        print("EXISTING ORIGINAL POINT CLOUD FOUND - SKIPPING SfM RECONSTRUCTION")
        print(f"{'='*60}")
        print(f"Using existing point cloud: {saved_original_cloud}")
        print("(Use --force-sfm to override this behavior)")
        point_cloud_path = str(saved_original_cloud)
    else:
        # Run SfM reconstruction
        print(f"\n{'='*60}")
        print("RUNNING SfM RECONSTRUCTION")
        print(f"{'='*60}")
        
        sfm_kwargs = {
            "voxel_size": args.voxel_size,  # For multi-scan alignment
        }
        
        if args.sfm_method == "multi":
            sfm_kwargs.update({
                "num_scans": args.num_scans,
                "overlap": args.overlap,
                "fitness_threshold": args.fitness_threshold,
            })
        
        point_cloud_path = run_sfm(
            input_images_dir, 
            sfm_output_dir, 
            method=args.sfm_method,
            **sfm_kwargs
        )
    
    # Run postprocessing (unless skipped)
    if args.skip_postprocess:
        print("\nSkipping postprocessing step")
        processed_cloud_path = point_cloud_path
    else:
        processed_cloud_path = output_dir / "processed_point_cloud.ply"
        
        # Determine COLMAP sparse directory for camera-based isolation
        colmap_sparse_dir = None
        if args.isolation_method == 'camera_centered':
            if args.sfm_method == "multi":
                # For multi-scan, use the first scan's sparse reconstruction
                # TODO: Could be improved to use combined camera poses from all scans
                colmap_sparse_dir = sfm_output_dir / "scan_1" / "reconstruction" / "sparse" / "0"
            else:
                # For single scan
                colmap_sparse_dir = sfm_output_dir / "sparse" / "0"
            
            # Check if the sparse directory exists
            if not colmap_sparse_dir.exists():
                print(f"Warning: COLMAP sparse directory not found at {colmap_sparse_dir}")
                print("Camera-based isolation methods require COLMAP reconstruction data.")
                print("Falling back to 'largest' isolation method.")
                args.isolation_method = 'largest'
                colmap_sparse_dir = None
        
        postprocess_kwargs = {
            "smart_downsample": args.smart_downsample,
            "downsample_factor": args.downsample_factor,
            "voxel_size": args.voxel_size,
            "clean": args.clean,
            "smooth": args.smooth,
            "smooth_radius": args.smooth_radius,
            "sigma_spatial": args.sigma_spatial,
            "sigma_normal": args.sigma_normal,
            "smooth_iterations": args.smooth_iterations,
            "isolate": args.isolate,
            "cluster_size_threshold": args.cluster_size_threshold,
            "top_k_clusters": args.top_k_clusters,
            "isolation_method": args.isolation_method,
            "remove_floor": args.remove_floor,
            "orient_by_floor": args.orient_by_floor,
            "normalize": args.normalize_postprocess,
            "crop_to_center": args.crop_to_center,
            "center_crop_factor": args.center_crop_factor,
            "colmap_sparse_dir": colmap_sparse_dir,
            "remove_white_flag": args.remove_white,
            "white_threshold": args.white_threshold,
        }
        
        run_postprocessing(point_cloud_path, processed_cloud_path, **postprocess_kwargs)
    
    # Run voxelization
    voxel_output_path = output_dir / "voxel_grid.npy"
    voxel_kwargs = {
        "voxel_size": args.voxel_size,
        "normalize": args.normalize_voxel,
    }
    
    if args.save_vis:
        voxel_kwargs["vis_output"] = output_dir / "voxel_visualizations"
    
    run_voxelization(processed_cloud_path, voxel_output_path, **voxel_kwargs)
    
    # Launch viewer if requested
    if args.view:
        run_model_viewer(voxel_output_path)
    
    # Cleanup intermediate files unless requested to keep them
    if not args.keep_intermediate:
        print("\nCleaning up intermediate files...")
        
        # Remove frames directory if we extracted from video
        if args.video and (output_dir / "frames").exists():
            shutil.rmtree(output_dir / "frames")
            print("  Removed extracted video frames")
        
        # Remove SfM intermediate files (keep only the final point cloud)
        if sfm_output_dir.exists():
            # Keep the final point cloud file
            final_cloud = Path(point_cloud_path)
            if final_cloud.exists():
                shutil.copy(final_cloud, output_dir / "original_point_cloud.ply")
            print("  Removed SfM intermediate files")
        
        print("  Cleanup completed")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Input: {'Video' if args.video else 'Images'} -> {args.video if args.video else args.images}")
    print(f"SfM Method: {args.sfm_method}")
    print(f"Postprocessing: {'Skipped' if args.skip_postprocess else 'Applied'}")
    print(f"Final voxel grid: {voxel_output_path}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Output directory: {output_dir}")
    
    if args.keep_intermediate:
        print(f"Intermediate files preserved in: {output_dir}")
    
    if args.view:
        print("Model viewer launched")
    
    print(f"{'='*60}")

if __name__ == '__main__':
    main()