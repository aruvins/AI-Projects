import fitz
import cv2
import numpy as np

from PIL import Image


def load_pdf(pdf_path):

    """
    Opens a PDF document.
    """

    pdf = fitz.open(pdf_path)

    return pdf


def pdf_page_count(pdf):

    """
    Returns number of pages.
    """

    return len(pdf)


def render_page_as_image(
    pdf,
    page_number,
    zoom=2
):

    """
    Renders a PDF page as an image.
    """

    page = pdf[page_number]

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pix = page.get_pixmap(
        matrix=matrix
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


def save_page_image(
    image,
    output_path
):

    """
    Saves rendered page image.
    """

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"Saved image: {output_path}"
    )


def extract_text_from_page(
    pdf,
    page_number
):

    """
    Extracts embedded text from PDF page.
    """

    page = pdf[page_number]

    text = page.get_text()

    return text


def convert_pdf_to_images(
    pdf_path,
    output_dir="outputs/"
):

    """
    Converts all PDF pages to images.
    """

    pdf = load_pdf(pdf_path)

    total_pages = pdf_page_count(pdf)

    print(
        f"Total Pages: {total_pages}"
    )

    for page_num in range(total_pages):

        image = render_page_as_image(
            pdf,
            page_num
        )

        output_path = (
            f"{output_dir}/page_{page_num}.png"
        )

        save_page_image(
            image,
            output_path
        )

    print("\nPDF conversion completed.")


def extract_all_text(pdf_path):

    """
    Extracts text from all pages.
    """

    pdf = load_pdf(pdf_path)

    full_text = ""

    for page_num in range(len(pdf)):

        text = extract_text_from_page(
            pdf,
            page_num
        )

        full_text += (
            f"\n--- Page {page_num} ---\n"
        )

        full_text += text

    return full_text