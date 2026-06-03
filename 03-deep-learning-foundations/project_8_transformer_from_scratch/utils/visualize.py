import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


# =====================================================
# TRAINING LOSS
# =====================================================

def plot_loss_curve(
    losses,
    save_path="outputs/training_loss.png"
):
    """
    Plot training loss vs epoch.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(losses) + 1),
        losses,
        marker="o"
    )

    plt.title(
        "Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


# =====================================================
# ATTENTION HEATMAP
# =====================================================

def plot_attention_heatmap(
    attention_matrix,
    save_path="outputs/attention_heatmap.png"
):
    """
    Visualize self-attention scores.

    Shape:
        [sequence_length, sequence_length]
    """

    plt.figure(figsize=(8, 6))

    plt.imshow(
        attention_matrix,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        "Self Attention Heatmap"
    )

    plt.xlabel(
        "Attended Word Position"
    )

    plt.ylabel(
        "Current Word Position"
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


# =====================================================
# POSITIONAL ENCODING
# =====================================================

def plot_positional_encoding(
    positional_encoding,
    save_path="outputs/positional_encoding.png"
):
    """
    Visualize sinusoidal positional encodings.

    Shape:
        [max_length, embedding_dim]
    """

    plt.figure(figsize=(10, 6))

    plt.imshow(
        positional_encoding,
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        "Positional Encoding Matrix"
    )

    plt.xlabel(
        "Embedding Dimension"
    )

    plt.ylabel(
        "Token Position"
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


# =====================================================
# WORD EMBEDDINGS PCA
# =====================================================

def plot_embedding_space(
    embedding_weights,
    vocab,
    num_words=100,
    save_path="outputs/embedding_space.png"
):
    """
    Visualize learned embeddings using PCA.

    embedding_weights:
        model.embedding.weight

    vocab:
        vocabulary dictionary
    """

    weights = (
        embedding_weights
        .detach()
        .cpu()
        .numpy()
    )

    words = list(vocab.keys())[:num_words]

    vectors = weights[:num_words]

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        vectors
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1]
    )

    for i, word in enumerate(words):

        plt.annotate(
            word,
            (
                reduced[i, 0],
                reduced[i, 1]
            )
        )

    plt.title(
        "Embedding Space (PCA)"
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()