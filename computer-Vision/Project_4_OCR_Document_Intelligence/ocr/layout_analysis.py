import cv2
import easyocr
import matplotlib.pyplot as plt


# Load image
image_path = "data/images/sample_document.png"

image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# Initialize EasyOCR
reader = easyocr.Reader(["en"])


# OCR results
results = reader.readtext(image_path)


# Draw bounding boxes
for result in results:

    bbox, text, confidence = result

    top_left = tuple(
        map(int, bbox[0])
    )

    bottom_right = tuple(
        map(int, bbox[2])
    )

    cv2.rectangle(
        image_rgb,
        top_left,
        bottom_right,
        (0, 255, 0),
        2
    )

    cv2.putText(
        image_rgb,
        text,
        (
            top_left[0],
            top_left[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


# Plot
plt.figure(figsize=(14, 10))

plt.imshow(image_rgb)

plt.title("OCR Layout Analysis")

plt.axis("off")

plt.show()