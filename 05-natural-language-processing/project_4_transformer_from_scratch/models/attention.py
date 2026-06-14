import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute the positional encodings using sine and cosine functions
        # The formula is based on the original Transformer paper by Vaswani et al. (2017)
        # For each dimension of the positional encoding, we use a different frequency of sine and cosine functions.
        # The div_term is calculated to ensure that the positional encodings have different frequencies for different dimensions.
        # The even dimensions use sine functions, while the odd dimensions use cosine functions.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        # The input x is expected to have shape (batch_size, seq_len, d_model)
        # We add the positional encoding to the input embeddings to provide information about the position of each token in the sequence.
        return x + self.pe[:, :x.size(1)]
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Compute the query, key, and value matrices
        # The input x is expected to have shape (batch_size, seq_len, embed_size)
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # Compute the attention scores using the dot product of the query and key matrices
        # The attention scores are then scaled by the square root of the embedding size to prevent large values that can lead to vanishing gradients.
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_size)

        # Apply softmax to get the attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute the output by multiplying the attention weights with the value matrix
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
        
        return output, attention_weights