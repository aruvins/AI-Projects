import fitz
import cv2
import numpy as np

from PIL import Image

# Open PDF
pdf = fitz.open(
    "data/pdfs/sample.pdf"
)

# Process pages
for page_num in range(len(pdf)):
    page = pdf[page_num]
    
    # Render page to image
    pix = page.get_pixmap()
    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Save image for OCR processing
    output_path = (
        f"outputs/page_{page_num}.png"
    )
    cv2.imwrite(output_path, img_cv)
    
    print(f"Page {page_num + 1} saved as {output_path}")