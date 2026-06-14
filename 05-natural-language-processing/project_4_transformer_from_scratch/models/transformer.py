import torch
import torch.nn as nn

from models.attention import PositionalEncoding
from models.transformer_block import TransformerBlock

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super().__init__()

        # Initialize the embedding layer, positional encoding, transformer blocks, and output layer
        # The embedding layer converts input token indices into dense vectors of a specified embedding size.
        # The positional encoding adds information about the position of each token in the sequence, which is crucial for the model to understand the order of tokens.
        # The transformer blocks consist of self-attention and feed-forward layers that allow the model to capture complex relationships between tokens in the input sequence.
        # The output layer is a linear layer that maps the output of the transformer blocks to the number of classes for classification tasks.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.transformer_blocks = TransformerBlock(embed_size)
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.positional_encoding(x)  # (batch_size, seq_len, embed_size)

        x, attention_weights = self.transformer_blocks(x)  # (batch_size, seq_len, embed_size)

        # Use the output of the first token (CLS token) for classification
        cls_token_output = x.mean(dim=1)  # (batch_size, embed_size)
        out = self.fc_out(cls_token_output)  # (batch_size, num_classes)
        
        return out, attention_weights