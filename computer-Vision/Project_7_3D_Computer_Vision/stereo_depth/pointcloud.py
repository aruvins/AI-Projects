import open3d as o3d
import numpy as np


def depth_to_pointcloud(depth_map, image, fx, fy, cx, cy):
    h, w = depth_map.shape

    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]

            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            colors.append(image[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd

def visualize_pointcloud(pcd):
    o3d.visualization.draw_geometries([pcd])