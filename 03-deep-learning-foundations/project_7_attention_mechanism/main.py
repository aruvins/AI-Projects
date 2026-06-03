import os
import torch

from utils.tokenizer import (
    tokenize,
    create_embeddings
)

from models.attention import SelfAttention

from utils.visualization import (
    plot_attention,
    plot_scores,
    plot_vectors
)

os.makedirs("outputs",exist_ok=True)

sentence = input("Please enter an example sentence: ")

print("\nSentence:")
print(sentence)

tokens = tokenize(sentence)
print("\nTokens:")
print(tokens)

# ---------------------
# Embeddings
# ---------------------
embeddings = create_embeddings(
    tokens,
    embedding_dim=8
)

print("\nEmbedding Shape:")
print(embeddings.shape)

# ---------------------
# Attention
# ---------------------

attention = SelfAttention(embedding_dim=8)

(
    context,
    weights,
    scores,
    Q,
    K,
    V
) = attention(embeddings)

# ---------------------
# Shapes
# ---------------------

print("\nQ Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

print("\nAttention Matrix Shape:")
print(weights.shape)

print("\nContext Shape:")
print(context.shape)

# ---------------------
# Visualization
# ---------------------

plot_attention(
    weights.detach().numpy(),
    tokens,
    "outputs/attention_heatmap.png"
)

plot_scores(
    scores.detach().numpy(),
    tokens,
    "outputs/raw_scores.png"
)

plot_vectors(
    Q.detach().numpy(),
    tokens,
    "Query Vectors",
    "outputs/query_vectors.png"
)

plot_vectors(
    K.detach().numpy(),
    tokens,
    "Key Vectors",
    "outputs/key_vectors.png"
)

plot_vectors(
    V.detach().numpy(),
    tokens,
    "Value Vectors",
    "outputs/value_vectors.png"
)

print("\nSaved visualizations:")
print(" outputs/attention_heatmap.png")
print(" outputs/raw_scores.png")
print(" outputs/query_vectors.png")
print(" outputs/key_vectors.png")
print(" outputs/value_vectors.png")