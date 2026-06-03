import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import TransformerClassifier

from utils.dataset import (
    build_vocab,
    create_dataloaders
)

from utils.trainer import (
    train_epoch,
    evaluate
)

from utils.predictor import (
    predict_review
)

from utils.visualize import (
    plot_loss_curve,
    plot_attention_heatmap,
    plot_positional_encoding
)

# =====================================================
# CONFIG
# =====================================================

VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_SIZE = 128

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_PATH = "outputs/transformer_model.pth"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# OUTPUT FOLDER
# =====================================================

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Transformer From Scratch")
print(f"Using device: {DEVICE}")
print("=" * 60)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading IMDB dataset...")

train_loader, test_loader, vocab = create_dataloaders(
    vocab_size=VOCAB_SIZE,
    max_length=MAX_LEN,
    batch_size=BATCH_SIZE
)

print(f"Vocabulary Size: {len(vocab)}")
print("Dataset Loaded\n")

# =====================================================
# BUILD MODEL
# =====================================================

model = TransformerClassifier(
    vocab_size=len(vocab),
    embed_size=EMBED_SIZE,
    max_len=MAX_LEN
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================================================
# LOAD OR TRAIN MODEL
# =====================================================

losses = []

if os.path.exists(MODEL_PATH):

    print("\nLoading saved model...")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    print("Model loaded successfully.")

else:

    print("\nNo saved model found. Training starting...\n")

    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )

        losses.append(train_loss)

        accuracy = evaluate(
            model,
            test_loader,
            DEVICE
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("\nTraining Complete")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# =====================================================
# LOSS VISUALIZATION
# =====================================================

print("\nGenerating Loss Plot...")

if len(losses) > 0:
    plot_loss_curve(
        losses,
        save_path="outputs/training_loss.png"
    )
else:
    print("Skipping loss plot (model was loaded, not trained).")

# =====================================================
# ATTENTION VISUALIZATION
# =====================================================

print("Generating Attention Heatmap...")

sample_batch = next(iter(test_loader))
sample_texts, sample_labels = sample_batch
sample_texts = sample_texts.to(DEVICE)

model.eval()

with torch.no_grad():
    _, attention = model(sample_texts[:1])

attention_matrix = (
    attention[0]
    .detach()
    .cpu()
    .numpy()
)

plot_attention_heatmap(
    attention_matrix,
    save_path="outputs/attention_heatmap.png"
)

# =====================================================
# POSITIONAL ENCODING VISUALIZATION
# =====================================================

print("Generating Positional Encoding Plot...")

positional_encoding = (
    model.position.pe[0]
    .detach()
    .cpu()
    .numpy()
)

plot_positional_encoding(
    positional_encoding,
    save_path="outputs/positional_encoding.png"
)

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

print("Generating Predictions...\n")

sample_reviews = [
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


results = []

for review in sample_reviews:

    prediction = predict_review(
        model=model,
        review=review,
        vocab=vocab,
        device=DEVICE,
        max_length=MAX_LEN
    )

    result = (
        f"Review:\n{review}\n"
        f"Prediction: {prediction}\n"
        f"{'-'*50}\n"
    )

    print(result)
    results.append(result)

with open("outputs/sample_predictions.txt", "w") as f:
    f.writelines(results)

# =====================================================
# SUMMARY
# =====================================================

print("\nFiles Generated:")
print("  outputs/attention_heatmap.png")
print("  outputs/positional_encoding.png")
print("  outputs/sample_predictions.txt")

if len(losses) > 0:
    print("  outputs/training_loss.png")

print("\nProject Complete.")