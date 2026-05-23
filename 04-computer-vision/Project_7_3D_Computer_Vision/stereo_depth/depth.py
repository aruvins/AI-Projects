import numpy as np


def disparity_to_depth(disparity_map, focal_length, baseline):
    disparity_map[disparity_map == 0] = 0.1

    depth_map = (focal_length * baseline) / disparity_map

    return depth_map