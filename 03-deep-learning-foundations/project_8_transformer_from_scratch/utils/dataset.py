import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


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

def build_vocab(texts, vocab_size=20000):
    """
    Build a vocabulary dictionary.

    Example:

    {
        "<PAD>": 0,
        "<UNK>": 1,
        "movie": 2,
        "great": 3,
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
# TEXT -> INTEGER TOKENS
# =====================================================

def encode_text(text,vocab, max_length):
    """
    Convert sentence to integer IDs.
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
# IMDB DATASET CLASS
# =====================================================

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab,max_length):
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

        label = self.labels[idx]

        return (
            torch.tensor(
                encoded_text,
                dtype=torch.long
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# =====================================================
# LOAD IMDB
# =====================================================

def load_imdb():

    print("Downloading IMDB dataset...")

    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print("Dataset loaded.")

    return (
        train_texts,
        train_labels,
        test_texts,
        test_labels
    )


# =====================================================
# DATALOADER CREATION
# =====================================================

def create_dataloaders(
    vocab_size=20000,
    max_length=200,
    batch_size=64
):
    """
    Main function called by main.py
    """

    (
        train_texts,
        train_labels,
        test_texts,
        test_labels
    ) = load_imdb()

    print("Building vocabulary...")

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