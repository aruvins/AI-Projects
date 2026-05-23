import cv2
import pytesseract

# Load image
image = cv2.imread(
    # "data/images/sample_document.jpg"
    "data/images/OCRtest.png"
)

# Convert to grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# OCR extraction
text = pytesseract.image_to_string(
    gray
)

print("Extracted Text:")
print(text)