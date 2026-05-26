import open3d as o3d
import numpy as np
import cv2

def depth_to_pointcloud(depth_map, image, fx, fy, cx, cy):
    # 1. Fix the OpenCV BGR to Open3D RGB color space mismatch
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = depth_map.shape

    # 2. Create a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 3. Create a mask to filter out invalid or zero depth values
    valid_mask = depth_map > 0
    
    z = depth_map[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]
    
    # 4. Vectorized 3D projection math
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into an (N, 3) array of float64 points
    points = np.stack((x, y, z), axis=1).astype(np.float64)
    
    # Extract corresponding normalized RGB colors
    colors = (image_rgb[valid_mask] / 255.0).astype(np.float64)

    # 5. Populate Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_pointcloud(pcd):
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Access the view control to manually position the visualizer's camera
    view_ctl = vis.get_view_control()
    
    # Look down the +Z direction from slightly behind the origin
    view_ctl.set_front([0, 0, -1])  # Direction the camera is looking
    view_ctl.set_up([0, -1, 0])     # Keeps your ground plane on the bottom (Y is down)
    view_ctl.set_lookat([0, 0, 5]) # Target point in front of the camera
    
    vis.run()
    vis.destroy_window()