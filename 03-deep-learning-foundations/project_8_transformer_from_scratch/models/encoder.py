import torch.nn as nn

from .attention import SelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion=4):
        super().__init__()
        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attention, weights = self.attention(x)
        x = self.norm1(x + attention)

        forward = self.feed_forward(x)
        out = self.norm2(x + forward)

        return out, weights