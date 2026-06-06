import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)


class BagOfWords:

    def __init__(self, vocab):
        self.vocab = vocab

    def transform(self, tokenized_texts):
        X = np.zeros(
            (
                len(tokenized_texts),
                len(self.vocab),
            ),
            dtype=np.float32,
        )

        unk_idx = self.vocab["<unk>"]

        for row, tokens in enumerate(tokenized_texts):

            counts = Counter(tokens)

            for word, count in counts.items():

                col = self.vocab.get(
                    word,
                    unk_idx,
                )

                X[row, col] = count

        return X
    



class TFIDF:
    def __init__(
        self,
        max_features=10000,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(
            texts
        )

    def transform(self, texts):
        return self.vectorizer.transform(
            texts
        )