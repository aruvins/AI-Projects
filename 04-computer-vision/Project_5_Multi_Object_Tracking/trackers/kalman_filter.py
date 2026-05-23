import numpy as np

from filterpy.kalman import KalmanFilter


def create_kalman_filter():

    """
    Creates Kalman Filter for motion tracking.
    """

    kf = KalmanFilter(
        dim_x=4,
        dim_z=2
    )

    kf.x = np.array([
        0,
        0,
        0,
        0
    ])

    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    kf.P *= 1000.

    kf.R = 5

    return kf