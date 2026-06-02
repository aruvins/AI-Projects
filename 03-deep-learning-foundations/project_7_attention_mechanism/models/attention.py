import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        # Define the linear layers for query, key, and value
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):
        """
        x shape:
        [tokens, embedding_dim]
        """
        
        # Compute query, key, and value
        Q = self.query(x)  # [batch_size, num_tokens, embedding_dim]
        K = self.key(x)      # [batch_size, num_tokens, embedding_dim]
        V = self.value(x)  # [batch_size, num_tokens, embedding_dim]
        
        # -------------------------
        # Attention Scores
        # -------------------------
        scores = torch.matmul(
            Q,
            K.T)
        
        # scaling factor to prevent large values in the softmax
        scale = K.shape[-1] ** 0.5
        scores = scores / scale

        # --------------------------
        # Softmax to get attention weights
        # --------------------------
        attention_weights = torch.softmax(
            scores, 
            dim=-1
        )
        
        # -------------------------
        # Context vectors
        # -------------------------

        context = torch.matmul(
            attention_weights,
            V
        )
        
        return (
            context,
            attention_weights,
            scores,
            Q,
            K,
            V
        )