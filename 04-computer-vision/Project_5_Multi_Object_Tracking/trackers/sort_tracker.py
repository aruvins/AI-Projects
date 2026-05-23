from deep_sort_realtime.deepsort_tracker import DeepSort


# Initialize tracker
tracker = DeepSort(
    max_age=30
)


def update_tracks(
    detections,
    frame
):

    """
    Update tracker with detections.
    """

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    return tracks