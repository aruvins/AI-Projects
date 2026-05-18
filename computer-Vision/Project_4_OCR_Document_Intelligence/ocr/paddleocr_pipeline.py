from paddleocr import PaddleOCR


ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="en"
)


results = ocr.predict(
    "data/images/sample_document.png"
)


print("\nEXTRACTED TEXT:\n")


for result in results:

    if "rec_texts" in result:

        texts = result["rec_texts"]

        scores = result["rec_scores"]

        for text, score in zip(texts, scores):

            print(f"Text: {text}")

            print(f"Confidence: {score:.4f}")

            print("-" * 30)