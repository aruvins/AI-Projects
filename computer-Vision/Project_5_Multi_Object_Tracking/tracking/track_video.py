import cv2
import os
import sys


sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from detectors.yolo_detector import (
    detect_objects
)

from trackers.sort_tracker import (
    update_tracks
)


# Video input
video_path = "data/videos/BasketBall.mp4"

cap = cv2.VideoCapture(video_path)


# Output writer
width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = int(
    cap.get(cv2.CAP_PROP_FPS)
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "outputs/tracked_output.mp4",
    fourcc,
    fps,
    (width, height)
)


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    # Object detection
    detections = detect_objects(frame)


    # Update tracker
    tracks = update_tracks(
        detections,
        frame
    )


    # Draw tracks
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(
            int,
            ltrb
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )


    # Write output
    out.write(frame)


    # Display
    cv2.imshow(
        "Multi-Object Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

out.release()

cv2.destroyAllWindows()