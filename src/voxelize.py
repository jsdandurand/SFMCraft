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

def main():
    parser = argparse.ArgumentParser(description="Convert point cloud to voxel grid")
    parser.add_argument('--input', type=str, required=True, help='Input PLY point cloud file')
    parser.add_argument('--output', type=str, help='Output path for voxel grid data (.npy)')
    parser.add_argument('--vis-output', type=str, help='Output directory for visualization files')
    parser.add_argument('--voxel-size', type=float, default=0.05, 
                       help='Size of voxels (default: 0.05)')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize point cloud to unit cube')
    
    args = parser.parse_args()

    try:
        # Load point cloud
        pcd = load_point_cloud(args.input)
        
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
        
        print(f"\n✓ Voxelization complete!")
        print(f"Final voxel grid: {grid.shape}")
        if args.output:
            print(f"Saved voxel data to: {args.output}")
        if args.vis_output:
            print(f"Saved visualizations to: {args.vis_output}")
        
    except Exception as e:
        print(f"Error during voxelization: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 