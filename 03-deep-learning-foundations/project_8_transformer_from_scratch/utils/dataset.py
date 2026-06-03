import os
import re
import tarfile
import urllib.request

from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


# =====================================================
# CONFIG
# =====================================================

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Reduce during development if desired
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):
    """
    Basic preprocessing.

    - lowercase
    - remove punctuation
    - split into words
    """

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    return text.split()


# =====================================================
# VOCABULARY
# =====================================================

def build_vocab(
    texts,
    vocab_size=20000
):
    """
    Build vocabulary.

    Example:

    {
        "<PAD>": 0,
        "<UNK>": 1,
        "movie": 2,
        ...
    }
    """

    counter = Counter()

    for text in texts:

        tokens = clean_text(text)

        counter.update(tokens)

    most_common = counter.most_common(
        vocab_size - 2
    )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for idx, (word, _) in enumerate(
        most_common,
        start=2
    ):
        vocab[word] = idx

    return vocab


# =====================================================
# ENCODING
# =====================================================

def encode_text(
    text,
    vocab,
    max_length
):
    """
    Convert text into token ids.
    """

    tokens = clean_text(text)

    encoded = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

    # truncate

    encoded = encoded[:max_length]

    # pad

    while len(encoded) < max_length:
        encoded.append(
            vocab["<PAD>"]
        )

    return encoded


# =====================================================
# DATASET
# =====================================================

class IMDBDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        vocab,
        max_length
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoded_text = encode_text(
            self.texts[idx],
            self.vocab,
            self.max_length
        )

        return (
            torch.tensor(
                encoded_text,
                dtype=torch.long
            ),
            torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        )


# =====================================================
# DOWNLOAD DATASET
# =====================================================

def download_and_extract(
    data_dir="data"
):
    """
    Download Stanford IMDB dataset.
    """

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    tar_path = os.path.join(
        data_dir,
        "aclImdb_v1.tar.gz"
    )

    extract_path = os.path.join(
        data_dir,
        "aclImdb"
    )

    if os.path.exists(extract_path):

        print(
            "Using existing dataset."
        )

        return extract_path

    print(
        "⬇ Downloading IMDB dataset..."
    )

    urllib.request.urlretrieve(
        URL,
        tar_path
    )

    print(
        "📦 Extracting dataset..."
    )

    with tarfile.open(
        tar_path,
        "r:gz"
    ) as tar:

        try:
            tar.extractall(
                path=data_dir,
                filter="data"
            )
        except TypeError:
            tar.extractall(
                path=data_dir
            )

    print("✅ Done.")

    return extract_path


# =====================================================
# LOAD IMDB
# =====================================================

def load_imdb():

    base_path = download_and_extract()

    train_texts = []
    train_labels = []

    test_texts = []
    test_labels = []

    # -------------------------
    # TRAIN
    # -------------------------

    for label in ["pos", "neg"]:

        folder = os.path.join(
            base_path,
            "train",
            label
        )

        for filename in os.listdir(folder):

            file_path = os.path.join(
                folder,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                train_texts.append(
                    f.read()
                )

                train_labels.append(
                    1 if label == "pos" else 0
                )

    # -------------------------
    # TEST
    # -------------------------

    for label in ["pos", "neg"]:

        folder = os.path.join(
            base_path,
            "test",
            label
        )

        for filename in os.listdir(folder):

            file_path = os.path.join(
                folder,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                test_texts.append(
                    f.read()
                )

                test_labels.append(
                    1 if label == "pos" else 0
                )

    # Optional smaller dataset
    if MAX_TRAIN_SAMPLES:

        train_texts = train_texts[
            :MAX_TRAIN_SAMPLES
        ]

        train_labels = train_labels[
            :MAX_TRAIN_SAMPLES
        ]

    if MAX_TEST_SAMPLES:

        test_texts = test_texts[
            :MAX_TEST_SAMPLES
        ]

        test_labels = test_labels[
            :MAX_TEST_SAMPLES
        ]

    return (
        train_texts,
        train_labels,
        test_texts,
        test_labels
    )


# =====================================================
# CREATE DATALOADERS
# =====================================================

def create_dataloaders(
    vocab_size=20000,
    max_length=200,
    batch_size=64
):

    (
        train_texts,
        train_labels,
        test_texts,
        test_labels
    ) = load_imdb()

    print(
        "Building vocabulary..."
    )

    vocab = build_vocab(
        train_texts,
        vocab_size
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    train_dataset = IMDBDataset(
        train_texts,
        train_labels,
        vocab,
        max_length
    )

    test_dataset = IMDBDataset(
        test_texts,
        test_labels,
        vocab,
        max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        test_loader,
        vocab
    )