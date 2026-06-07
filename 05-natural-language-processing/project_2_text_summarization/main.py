from utils.preprocess import (
    preprocess_text,
    split_sentences
)

from models.extractive import summarize
from models.transformer import (
    summarize as transformer_summary
)

with open(
    "data/article.txt",
    "r"
) as f:

    article = f.read()

clean_text = preprocess_text(article)

sentences = split_sentences(clean_text)

extractive_summary = summarize(
    sentences,
    num_sentences=3
)

print("\nEXTRACTIVE SUMMARY\n")
print(extractive_summary)

print("\nTRANSFORMER SUMMARY\n")
print(transformer_summary(article))