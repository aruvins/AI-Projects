import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


def tokenize(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    return text.split()


def build_vocab(texts, max_vocab_size=10000):

    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    most_common = counter.most_common(
        max_vocab_size - 2
    )

    for idx, (word, _) in enumerate(
        most_common,
        start=2
    ):
        vocab[word] = idx

    return vocab


def encode_text(
    text,
    vocab,
    max_length=128
):

    tokens = tokenize(text)

    encoded = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

    encoded = encoded[:max_length]

    encoded += [0] * (
        max_length - len(encoded)
    )

    return encoded


class IMDBDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        vocab,
        max_length=128
    ):

        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        x = encode_text(
            self.texts[idx],
            self.vocab,
            self.max_length
        )

        y = self.labels[idx]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )


def create_dataloaders(
    batch_size=32,
    max_length=128
):

    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    vocab = build_vocab(train_texts)

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
        batch_size=batch_size
    )

    return (
        train_loader,
        test_loader,
        vocab
    )