import os
import tarfile
import urllib.request
from collections import Counter

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def tokenize(text):
    return text.lower().split()


def build_vocab(tokenized_texts, min_freq=2):
    counter = Counter()

    for text in tokenized_texts:
        counter.update(text)

    vocab = {"<pad>": 0, "<unk>": 1}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def download_and_extract(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)

    tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extract_path = os.path.join(data_dir, "aclImdb")

    if os.path.exists(extract_path):
        return extract_path

    print("⬇ Downloading IMDB dataset...")
    urllib.request.urlretrieve(URL, tar_path)

    print("📦 Extracting dataset...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print("✅ Done.")

    return extract_path


def load_imdb():
    base_path = download_and_extract()

    train_texts = []
    train_tokens = []
    train_labels = []

    test_texts = []
    test_tokens = []
    test_labels = []

    for label in ["pos", "neg"]:

        folder = os.path.join(
            base_path,
            "train",
            label,
        )

        for file in sorted(
            os.listdir(folder)
        ):

            with open(
                os.path.join(folder, file),
                "r",
                encoding="utf-8",
            ) as f:

                text = f.read()

                tokens = tokenize(text)

                train_texts.append(text)
                train_tokens.append(tokens)

                train_labels.append(
                    1 if label == "pos" else 0
                )

    for label in ["pos", "neg"]:

        folder = os.path.join(
            base_path,
            "test",
            label,
        )

        for file in sorted(
            os.listdir(folder)
        ):

            with open(
                os.path.join(folder, file),
                "r",
                encoding="utf-8",
            ) as f:

                text = f.read()

                tokens = tokenize(text)

                test_texts.append(text)
                test_tokens.append(tokens)

                test_labels.append(
                    1 if label == "pos" else 0
                )

    vocab = build_vocab(train_tokens)

    return (
        train_texts,
        train_tokens,
        train_labels,
        test_texts,
        test_tokens,
        test_labels,
        vocab,
    )