import matplotlib.pyplot as plt
import seaborn as sns


def plot_matrix(
    matrix,
    title,
    save_path,
    xlabels=None,
    ylabels=None,
    cmap="Blues"
):
    """
    Generic heatmap plotter.
    """

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=xlabels,
        yticklabels=ylabels
    )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def plot_attention(
    attention_matrix,
    tokens,
    save_path
):
    plot_matrix(
        attention_matrix,
        title="Attention Weights",
        save_path=save_path,
        xlabels=tokens,
        ylabels=tokens
    )


def plot_scores(
    scores,
    tokens,
    save_path
):
    plot_matrix(
        scores,
        title="Raw Attention Scores (QKᵀ)",
        save_path=save_path,
        xlabels=tokens,
        ylabels=tokens,
        cmap="Reds"
    )


def plot_vectors(
    vectors,
    tokens,
    title,
    save_path
):
    embedding_dims = [
        f"d{i}"
        for i in range(vectors.shape[1])
    ]

    plot_matrix(
        vectors,
        title=title,
        save_path=save_path,
        xlabels=embedding_dims,
        ylabels=tokens,
        cmap="viridis"
    )