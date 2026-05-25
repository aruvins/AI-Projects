import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.download_data import download_dataset, extract_dataset, build_csv, fallback_synthetic

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 0. Prepare dataset
# ----------------------------
try:
    download_dataset()
    extract_dataset()
    build_csv()
except Exception as e:
    print("❌ Error occurred during dataset preparation:", e)
    fallback_synthetic()

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("data/spam.csv")  # change path if needed

X = df["text"]
y = df["label"]  # "spam" or "ham"


# Convert labels to binary
y = y.map({"ham": 0, "spam": 1})

plt.figure()
sns.countplot(data=df, x="label")
plt.title("Spam vs Ham Distribution")
plt.show()


# ----------------------------
# 2. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# 3. Choose model
# ----------------------------
models = {
    "naive_bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")), #TF-IDF vectorizer to convert text to numerical features
        ("clf", MultinomialNB()) # Classifier to predict spam vs ham
    ]),
    "logistic_regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")), #TF-IDF vectorizer to convert text to numerical features
        ("clf", LogisticRegression(max_iter=1000)) # Classifier to predict spam vs ham
    ])
}


for name, model in models.items():
    print(f"\n==============================")
    print(f"Training model: {name}")
    print(f"==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ----------------------------
    # Metrics
    # ----------------------------
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print(report)

    # ----------------------------
    # Create model output folder
    # ----------------------------
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # ----------------------------
    # Save classification report
    # ----------------------------
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # ----------------------------
    # Save predictions on sample data
    # ----------------------------
    samples = [
        "Congratulations! You won a free iPhone. Click here now!",
        "Hey, are we still on for dinner tonight?",
        "Urgent: your account has been suspended, verify immediately"
    ]

    preds = model.predict(samples)

    with open(os.path.join(model_dir, "sample_predictions.txt"), "w") as f:
        for text, pred in zip(samples, preds):
            label = "SPAM" if pred == 1 else "HAM"
            f.write(f"{label}: {text}\n")

    # ----------------------------
    # Save confusion matrix plot
    # ----------------------------
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plot_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"📁 Saved outputs to: {model_dir}")

# ----------------------------
# 6. Try custom predictions
# ----------------------------
samples = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, are we still on for dinner tonight?",
    "Urgent: your account has been suspended, verify immediately"
]

preds = model.predict(samples)

for text, pred in zip(samples, preds):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{label}: {text}")