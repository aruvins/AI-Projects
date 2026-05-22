import cv2
import matplotlib.pyplot as plt

from disparity import compute_disparity
from depth import disparity_to_depth
from pointcloud import depth_to_pointcloud, visualize_pointcloud

left = cv2.imread('../data/stereo/left.png')
right = cv2.imread('../data/stereo/right.png')


disparity = compute_disparity(left, right)

plt.imshow(disparity, cmap='plasma')
plt.title('Disparity Map')
plt.colorbar()
plt.show()

focal_length = 721.0
baseline = 0.54

depth = disparity_to_depth(disparity, focal_length, baseline)

pcd = depth_to_pointcloud(
    depth,
    left,
    fx=721.0,
    fy=721.0,
    cx=left.shape[1] / 2,
    cy=left.shape[0] / 2
)

visualize_pointcloud(pcd)