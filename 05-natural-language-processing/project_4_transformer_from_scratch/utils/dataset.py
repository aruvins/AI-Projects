import torch

from torch.utils.data import (
    Dataset,
    DataLoader
)

from utils.imdb_loader import load_imdb


def encode_tokens(
    tokens,
    vocab,
    max_length=256
):
    encoded = [
        vocab.get(
            token,
            vocab["<unk>"]
        )
        for token in tokens
    ]

    encoded = encoded[:max_length]

    encoded += [vocab["<pad>"]] * (
        max_length - len(encoded)
    )

    return encoded


class IMDBDataset(Dataset):

    def __init__(
        self,
        tokenized_texts,
        labels,
        vocab,
        max_length=256
    ):

        self.texts = tokenized_texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoded = encode_tokens(
            self.texts[idx],
            self.vocab,
            self.max_length
        )

        label = self.labels[idx]

        return (
            torch.tensor(
                encoded,
                dtype=torch.long
            ),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


def create_dataloaders(
    batch_size=32,
    max_length=256
):

    (
        train_texts,
        train_tokens,
        train_labels,
        test_texts,
        test_tokens,
        test_labels,
        vocab,
    ) = load_imdb()

    train_dataset = IMDBDataset(
        train_tokens,
        train_labels,
        vocab,
        max_length
    )

    test_dataset = IMDBDataset(
        test_tokens,
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