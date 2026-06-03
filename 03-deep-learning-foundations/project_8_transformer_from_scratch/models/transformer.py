import torch
import torch.nn as nn

from .encoder import EncoderBlock
from .positional_encoding import PositionalEncoding

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=128, max_len=200):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEncoding(embed_size, max_len)
        self.encoder = EncoderBlock(embed_size)
        self.fc = nn.Linear(embed_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x, attention = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.fc(x)

        return logits, attention