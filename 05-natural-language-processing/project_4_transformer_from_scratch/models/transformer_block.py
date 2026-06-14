import torch.nn as nn
from models.attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        # Initialize the self-attention mechanism, layer normalization, and feed-forward network
        # The self-attention mechanism allows the model to focus on different parts of the input sequence when processing each token.
        # Layer normalization is used to stabilize the training process and improve convergence.
        # The feed-forward network is a simple two-layer fully connected network that processes the output of the self-attention mechanism.
        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # The feed-forward network consists of two linear layers with a ReLU activation in between. The hidden layer has a size of 4 times the embedding size, which is a common choice in Transformer architectures.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )

    def forward(self, x):
        # Apply self-attention and add the residual connection
        attention_output, attention_weights = self.attention(x)
        x = self.norm1(x + attention_output)

        # Apply the feed-forward network and add the residual connection
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + feed_forward_output)

        return x, attention_weights