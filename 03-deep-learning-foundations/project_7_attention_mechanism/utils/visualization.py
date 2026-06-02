import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_attention(attention_matrix,tokens,save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        cmap="Blues"
    )
    plt.title("Attention Weights")
    plt.xlabel("Attended To")
    plt.ylabel("Current Token")
    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()