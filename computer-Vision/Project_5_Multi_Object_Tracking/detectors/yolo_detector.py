import cv2

from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolov8n.pt")


def detect_objects(frame):

    """
    Detect objects using YOLO.
    """

    results = model(frame)[0]

    detections = []

    for result in results.boxes:

        x1, y1, x2, y2 = (
            result.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = (
            result.conf[0]
            .cpu()
            .numpy()
        )

        class_id = int(
            result.cls[0]
            .cpu()
            .numpy()
        )

        detections.append([
            [x1, y1, x2, y2],
            confidence,
            class_id
        ])

    return detections