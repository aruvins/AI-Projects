import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.queries(x)
        K = self.keys(x)
        V = self.values(x)

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores /
            math.sqrt(self.embed_size)
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights