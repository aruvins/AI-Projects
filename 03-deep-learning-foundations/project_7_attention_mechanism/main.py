import os
import torch

from utils.tokenizer import (
    tokenize,
    create_embeddings
)

from models.attention import SelfAttention

from utils.visualization import (
    plot_attention
)

os.makedirs("outputs",exist_ok=True)

EXAMPLE_SENTENCES = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "attention helps models focus",
]

sentence = EXAMPLE_SENTENCES[0]

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

print(
    "\nSaved:"
    " outputs/attention_heatmap.png"
)