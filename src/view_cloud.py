import open3d as o3d
import numpy as np
import argparse

def normalize_points(points):
    """Center and normalize points to unit cube"""
    # Center
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit cube
    scale = np.max(np.abs(points))
    points = points / scale
    
    return points

def view_point_cloud(ply_path):
    # Load the point cloud
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points")
    
    # Normalize points
    points = normalize_points(points)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coord_frame],
                                    window_name="Point Cloud Viewer",
                                    width=1024,
                                    height=768,
                                    point_show_normal=False)

def main():
    parser = argparse.ArgumentParser(description="Interactive point cloud viewer")
    parser.add_argument('ply_file', help='Path to the PLY file to visualize')
    args = parser.parse_args()
    
    view_point_cloud(args.ply_file)

if __name__ == '__main__':
    main() 