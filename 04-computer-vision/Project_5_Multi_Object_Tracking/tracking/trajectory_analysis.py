import cv2
import numpy as np


# Store trajectories
trajectories = {}


def update_trajectory(
    track_id,
    center
):

    if track_id not in trajectories:

        trajectories[track_id] = []

    trajectories[track_id].append(center)


def draw_trajectories(
    frame
):

    for track_id, points in trajectories.items():

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (0, 255, 255),
                2
            )

    return frame