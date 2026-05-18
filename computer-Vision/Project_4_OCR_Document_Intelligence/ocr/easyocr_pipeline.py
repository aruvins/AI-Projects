import cv2
import easyocr

# Initialize reader
reader = easyocr.Reader(['en'])

# Load image
image_path = "data/images/sample_document.jpg"

# OCR
results = reader.readtext(image_path)

# Print results
for result in results:
    bbox, text, confidence = result
    
    print(f"Text: {text}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)