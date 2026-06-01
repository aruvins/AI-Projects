import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.vocabulary import build_vocab
from data.dataset import create_dataloader
from utils.trainer import train_epoch, train_model
from utils.visualize import plot_metrics, compare_models
from utils.predictor import predict

from models.lstm_model import LSTMClassifier
from models.rnn_model import RNNClassifier

# Make output directory and set up training device
os.makedirs("outputs",exist_ok=True)
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Example texts and labels (1 is a positive review and 0 is negative) -> Labels are the correct answer
texts = [
    "this movie was amazing",
    "i loved this film",
    "terrible movie",
    "worst acting ever",
    "fantastic story",
    "bad film"
]

labels = [
    1,
    1,
    0,
    0,
    1,
    0
]

# load vocab
vocab = build_vocab(texts)

# download training data
loader = create_dataloader(
    texts,
    labels,
    vocab
)

# --------------------------- RNN Model ----------------------------
rnn_model = RNNClassifier(
    vocab_size=len(vocab)
).to(DEVICE)


print("\nTraining RNN...\n")
rnn_losses, rnn_accuracies = train_model(
    rnn_model,
    loader,
    DEVICE
)

plot_metrics(
    "RNN Model",
    rnn_losses,
    rnn_accuracies
)

# --------------------------- LSTM Model ---------------------------
lstm_model = LSTMClassifier(
    vocab_size=len(vocab)
).to(DEVICE)

print("\nTraining LSTM...\n")
lstm_losses, lstm_accuracies = train_model(
    lstm_model,
    loader,
    DEVICE
)
plot_metrics(
    "LSTM Model",
    rnn_losses,
    rnn_accuracies
)

# Compare LSTM vs RNN
compare_models(
    rnn_losses,
    lstm_losses,
    rnn_accuracies,
    lstm_accuracies
)
# --------------------------------------------------------------------------
# --------------------------- Sample Predictions ---------------------------
# --------------------------------------------------------------------------

test_sentences = [
    # Very Positive
    "This was one of the best movies I have ever seen",
    "Absolutely incredible acting and a fantastic story",
    "I loved every minute of this film",
    "The performances were outstanding and the ending was perfect",

    # Positive
    "A fun movie with strong characters and good pacing",
    "I enjoyed the film and would watch it again",
    "The story was engaging and the visuals looked great",

    # Neutral / Mixed
    "The movie was okay but nothing particularly memorable",
    "Some parts were entertaining while others felt slow",
    "The acting was decent but the plot was average",
    "It was neither good nor bad, just an ordinary movie",

    # Negative
    "The story dragged on and several scenes felt unnecessary",
    "I expected more from the movie and left disappointed",
    "The acting was weak and the dialogue felt unnatural",

    # Very Negative
    "One of the worst films I have ever watched",
    "The plot made no sense and the acting was terrible",
    "A complete waste of time from beginning to end"
]

with open("outputs/predictions.txt", "w") as f:

    f.write("RNN vs LSTM Predictions\n")
    f.write("=" * 50 + "\n")

    for sentence in test_sentences:
        rnn_pred, rnn_confidence = predict(sentence, rnn_model, vocab, DEVICE)
        lstm_pred, lstm_confidence  = predict(sentence, lstm_model, vocab, DEVICE)

        f.write(f"Sentence: {sentence}\n")
        f.write(f"RNN Prediction : {rnn_pred}\n")
        f.write(f"RNN Confidence : {rnn_confidence}\n")
        f.write(f"\n")
        f.write(f"LSTM Prediction: {lstm_pred}\n")
        f.write(f"LSTM Confidence : {lstm_confidence}\n")
        f.write("-" * 50 + "\n")

print("Predictions saved to outputs/predictions.txt")