import os

import matplotlib.pyplot as plt

from utils.imdb_loader import load_imdb
from utils.vectorizers import (
    BagOfWords,
    TFIDF,
)

from models.logistic_regression import (
    TextClassifier,
)

from utils.metrics import evaluate


# ==========================================
# Configuration
# ==========================================

USE_TFIDF = True

# ==========================================
# Setup
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True,
)

print("Loading IMDB dataset...")

(
    train_texts,
    train_tokens,
    train_labels,
    test_texts,
    test_tokens,
    test_labels,
    vocab,
) = load_imdb()

print(
    f"Training reviews: {len(train_texts)}"
)

print(
    f"Testing reviews: {len(test_texts)}"
)

print(
    f"Vocabulary size: {len(vocab)}"
)

# ==========================================
# Feature Extraction
# ==========================================

if USE_TFIDF:

    print("\nUsing TF-IDF Features...")

    vectorizer = TFIDF(
        max_features=10000
    )

    X_train = vectorizer.fit_transform(
        train_texts
    )

    X_test = vectorizer.transform(
        test_texts
    )

else:

    print("\nUsing Bag-of-Words Features...")

    vectorizer = BagOfWords(
        vocab
    )

    X_train = vectorizer.transform(
        train_tokens
    )

    X_test = vectorizer.transform(
        test_tokens
    )

print(
    f"Feature shape: {X_train.shape}"
)

# ==========================================
# Train Model
# ==========================================

print("\nTraining model...")

model = TextClassifier()

model.fit(
    X_train,
    train_labels,
)

# ==========================================
# Evaluate
# ==========================================

print("\nEvaluating...")

predictions = model.predict(
    X_test
)

cm = evaluate(
    test_labels,
    predictions,
)

# ==========================================
# Confusion Matrix
# ==========================================

plt.figure(
    figsize=(6, 6)
)

plt.imshow(cm)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

print(
    "\nSaved confusion matrix."
)

# ==========================================
# Interactive Predictions
# ==========================================

while True:

    review = input(
        "\nEnter review (or 'quit'): "
    )

    if review.lower() == "quit":
        break

    if USE_TFIDF:

        x = vectorizer.transform(
            [review]
        )

    else:

        tokens = review.lower().split()

        x = vectorizer.transform(
            [tokens]
        )

    prediction = model.predict(x)[0]

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    print(
        f"Prediction: {sentiment}"
    )