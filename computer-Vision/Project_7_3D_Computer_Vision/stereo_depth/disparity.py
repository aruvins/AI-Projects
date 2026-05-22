import cv2
import numpy as np
from tqdm import tqdm


def compute_disparity(left_img, right_img, max_disparity=64, block_size=7):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    h, w = left_gray.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    half_block = block_size // 2

    for y in tqdm(range(half_block, h - half_block)):
        for x in range(half_block, w - half_block):

            best_disparity = 0
            min_error = float('inf')

            left_block = left_gray[
                y-half_block:y+half_block+1,
                x-half_block:x+half_block+1
            ]

            for d in range(max_disparity):
                xr = x - d

                if xr - half_block < 0:
                    continue

                right_block = right_gray[
                    y-half_block:y+half_block+1,
                    xr-half_block:xr+half_block+1
                ]

                error = np.sum((left_block.astype(np.float32) -
                                right_block.astype(np.float32)) ** 2)

                if error < min_error:
                    min_error = error
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map