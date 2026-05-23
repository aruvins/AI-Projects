import cv2
import matplotlib.pyplot as plt
import numpy as np

from disparity import compute_disparity
from depth import disparity_to_depth
from pointcloud import depth_to_pointcloud, visualize_pointcloud

import cv2
import glob
import os

# 1. Get the directory where run_stereo.py actually lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the absolute path to your data folder
# (Moving up one level from 'stereo_depth' to the root project folder, then into 'data')
stereo_dir = os.path.abspath(os.path.join(script_dir, "../data/stereo/kitti/training"))

# Debug print to confirm exactly where Python is looking
print(f"Searching for images in: {stereo_dir}")

# 3. KITTI conventions: image_2 is left color, image_3 is right color
left_images = sorted(glob.glob(os.path.join(stereo_dir, "image_2", "*.png")))
right_images = sorted(glob.glob(os.path.join(stereo_dir, "image_3", "*.png")))

if len(left_images) == 0 or len(right_images) == 0:
    raise ValueError(f"No stereo image pairs found. Check dataset structure.\n"
                     f"Found {len(left_images)} left images and {len(right_images)} right images.")

left = cv2.imread(left_images[0])
right = cv2.imread(right_images[0])

if left is None or right is None:
    raise ValueError("Failed to load images. Check file paths.")

disparity = compute_disparity(left, right)

plt.imshow(disparity, cmap='plasma')
plt.title('Disparity Map')
plt.colorbar()
plt.show()

focal_length = 721.0
baseline = 0.54

# 1. Compute the raw depth map
depth = disparity_to_depth(disparity, focal_length, baseline)

# 2. Create a mask for valid depth ranges (e.g., between 0.5m and 80m)
min_depth = 0.5
max_depth = 80.0
valid_mask = (depth > min_depth) & (depth < max_depth) & (disparity > 0)

# 3. Clean up the depth map and the color image using the mask
filtered_depth = np.where(valid_mask, depth, 0)

# Real KITTI camera intrinsic parameters
kitti_fx = 721.0
kitti_fy = 721.0
kitti_cx = 609.55
kitti_cy = 172.85

pcd = depth_to_pointcloud(
    filtered_depth,
    left,
    fx=kitti_fx,
    fy=kitti_fy,
    cx=kitti_cx,
    cy=kitti_cy
)

visualize_pointcloud(pcd)