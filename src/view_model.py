import open3d as o3d
import numpy as np
import argparse
import os

def load_point_cloud(file_path):
    """Load and preprocess a point cloud from a PLY file."""
    print(f"Loading point cloud from {file_path}...")
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud contains {len(pcd.points)} points.")
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

def load_voxel_grid(file_path):
    """Load a voxel grid from a NPY file.
    The grid is a 4D array (x, y, z, channels) where channels are [occupancy, R, G, B]
    """
    print(f"Loading voxel grid from {file_path}...")
    if not file_path.endswith('.npy'):
        raise ValueError("Voxel grid must be a .npy file")
    
    grid = np.load(file_path)
    print(f"Voxel grid shape: {grid.shape}")
    return grid

def create_voxel_mesh(grid, voxel_size=1.0, offset=[0, 0, 0]):
    """Convert a voxel grid (4D numpy array) to an Open3D mesh for visualization."""
    # Find occupied voxels
    voxels = np.argwhere(grid[..., 0] > 0)
    
    # Create mesh for all voxels
    mesh_boxes = []
    for voxel in voxels:
        # Create a box for each voxel
        box = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        # Get color for this voxel
        color = grid[voxel[0], voxel[1], voxel[2], 1:]  # RGB values
        box.paint_uniform_color(color)
        # Move box to correct position
        box.translate([
            (voxel[0] + offset[0]) * voxel_size,
            (voxel[1] + offset[1]) * voxel_size,
            (voxel[2] + offset[2]) * voxel_size
        ])
        mesh_boxes.append(box)
    
    # Combine all boxes into a single mesh
    if mesh_boxes:
        combined_mesh = mesh_boxes[0]
        for box in mesh_boxes[1:]:
            combined_mesh += box
        return combined_mesh
    return None

def visualize_geometries(geometries, window_name="Open3D Viewer"):
    """Visualize multiple geometries in Open3D."""
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name, width=1280, height=720)
    
    # Add all geometries to the viewer
    for geom in geometries:
        if geom is not None:
            viewer.add_geometry(geom)
    
    # Set default camera view
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 2.0
    
    # Update visualization
    viewer.update_renderer()
    viewer.poll_events()
    viewer.update_renderer()
    
    # Run the visualizer
    viewer.run()
    viewer.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds and voxel grids")
    parser.add_argument("--point_cloud", type=str, help="Path to point cloud PLY file")
    parser.add_argument("--voxel_grid", type=str, help="Path to voxel grid NPY file")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Size of each voxel for visualization")
    parser.add_argument("--offset", type=float, nargs=3, default=[0, 0, 0], help="Offset for voxel grid visualization [x y z]")
    parser.add_argument("--crop-factor", type=float, default=None,
                       help="Factor to crop central region for point cloud (0-1, e.g., 0.8 = central 80%%)")
    args = parser.parse_args()

    if not args.point_cloud and not args.voxel_grid:
        parser.error("At least one of --point_cloud or --voxel_grid must be provided")

    geometries = []

    # Load point cloud if specified
    if args.point_cloud:
        if not os.path.exists(args.point_cloud):
            print(f"Error: Point cloud file {args.point_cloud} does not exist")
            return
        pcd = load_point_cloud(args.point_cloud)
        
        # Crop point cloud if requested
        if args.crop_factor is not None:
            if args.crop_factor <= 0 or args.crop_factor > 1:
                print("Error: crop-factor must be between 0 and 1")
                return
            print(f"\nCropping to central {args.crop_factor * 100}% of the scene...")
            pcd = crop_to_center(pcd, args.crop_factor)
        
        geometries.append(pcd)

    # Load and convert voxel grid if specified
    if args.voxel_grid:
        if not os.path.exists(args.voxel_grid):
            print(f"Error: Voxel grid file {args.voxel_grid} does not exist")
            return
        voxel_grid = load_voxel_grid(args.voxel_grid)
        voxel_mesh = create_voxel_mesh(voxel_grid, args.voxel_size, args.offset)
        if voxel_mesh is not None:
            geometries.append(voxel_mesh)

    if not geometries:
        print("No valid geometries to display")
        return

    # Visualize the geometries
    visualize_geometries(geometries)

if __name__ == "__main__":
    main() 