import torch
import torch.nn as nn
import math
from torch import Tensor


class PositionalEncoding(nn.Module):

    pe: Tensor

    def __init__(self, embed_size, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, embed_size)

        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embed_size,
                2,
                dtype=torch.float
            )
            * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]