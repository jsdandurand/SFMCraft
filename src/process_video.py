#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print its output in real-time."""
    print(f"\n=== {description} ===")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process video to create voxelized 3D model")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Directory to store output files (default: output)")
    parser.add_argument("--n_images", type=int, default=50,
                       help="Number of images to sample from the video (default: 50)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                       help="Size of voxels in the output grid (default: 0.1)")
    parser.add_argument("--clean", action="store_true",
                       help="Remove outliers from point cloud")
    parser.add_argument("--remove-floor", action="store_true",
                       help="Remove floor points from point cloud")
    parser.add_argument("--isolate", action="store_true",
                       help="Isolate main object from point cloud")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize point cloud to unit cube")
    parser.add_argument("--view", action="store_true",
                       help="View the resulting point cloud and voxel grid")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base names for output files
    video_name = Path(args.video).stem
    frames_dir = output_dir / f"{video_name}_frames"
    ply_output = output_dir / f"{video_name}_model.ply"
    voxel_output = output_dir / f"{video_name}_voxels.npy"

    # Step 1: Sample video frames
    sample_cmd = f"python src/sample_video_frames.py --video {args.video} --n_images {args.n_images} --output {frames_dir}"
    run_command(sample_cmd, "Sampling video frames")

    # Step 2: Run SFM
    sfm_cmd = f"python src/sfm.py --input {frames_dir} --output {ply_output} --orient"
    run_command(sfm_cmd, "Running Structure from Motion")

    # Step 3: Voxelize the point cloud
    voxelize_cmd = f"python src/voxelize.py --input {ply_output} --output {voxel_output} --voxel-size {args.voxel_size}"
    if args.clean:
        voxelize_cmd += " --clean"
    if args.remove_floor:
        voxelize_cmd += " --remove-floor"
    if args.normalize:
        voxelize_cmd += " --normalize"
    if args.isolate:
        voxelize_cmd += " --isolate"
    run_command(voxelize_cmd, "Voxelizing point cloud")

    # Step 4: View results if requested
    if args.view:
        view_cmd = f"python src/view_model.py --point_cloud {ply_output} --voxel_grid {voxel_output}"
        run_command(view_cmd, "Viewing results")

    print("\n=== Processing Complete ===")
    print(f"Point cloud saved to: {ply_output}")
    print(f"Voxel grid saved to: {voxel_output}")

if __name__ == "__main__":
    main() 