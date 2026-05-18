import cv2
import matplotlib.pyplot as plt


def show_image(
    image,
    title="Image",
    figsize=(10, 10)
):

    """
    Displays image using matplotlib.
    """

    plt.figure(figsize=figsize)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(image_rgb)

    plt.title(title)

    plt.axis("off")

    plt.show()


def draw_ocr_boxes(
    image,
    results,
    color=(0, 255, 0)
):

    """
    Draw OCR bounding boxes.

    Compatible with EasyOCR output.
    """

    image_copy = image.copy()

    for result in results:

        bbox, text, confidence = result

        top_left = tuple(
            map(int, bbox[0])
        )

        bottom_right = tuple(
            map(int, bbox[2])
        )

        cv2.rectangle(
            image_copy,
            top_left,
            bottom_right,
            color,
            2
        )

        cv2.putText(
            image_copy,
            text,
            (
                top_left[0],
                top_left[1] - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image_copy


def draw_paddleocr_boxes(
    image,
    results,
    color=(255, 0, 0)
):

    """
    Draw bounding boxes from PaddleOCR.
    """

    image_copy = image.copy()

    for line in results[0]:

        points = line[0]

        text = line[1][0]

        pt1 = tuple(
            map(int, points[0])
        )

        pt3 = tuple(
            map(int, points[2])
        )

        cv2.rectangle(
            image_copy,
            pt1,
            pt3,
            color,
            2
        )

        cv2.putText(
            image_copy,
            text,
            (
                pt1[0],
                pt1[1] - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image_copy


def draw_layout_boxes(
    image,
    layout,
    color=(0, 0, 255)
):

    """
    Draw layout detection regions.
    """

    image_copy = image.copy()

    for block in layout:

        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)

        x2 = int(block.block.x_2)
        y2 = int(block.block.y_2)

        label = block.type

        cv2.rectangle(
            image_copy,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            image_copy,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return image_copy


def visualize_table_regions(
    image,
    tables,
    color=(255, 255, 0)
):

    """
    Visualize detected table regions.
    """

    image_copy = image.copy()

    for table in tables:

        x1, y1, x2, y2 = table

        cv2.rectangle(
            image_copy,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

    return image_copy