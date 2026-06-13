import json
import random

import nltk
import torch

from utils.bag_of_words import tokenize
from utils.bag_of_words import bag_of_words

from utils.model import ChatbotNN


# Download tokenizer
nltk.download("punkt")


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

data = torch.load(
    "outputs/chatbot_model.pth"
)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]


# --------------------------------------------------
# Rebuild Model
# --------------------------------------------------

model = ChatbotNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size
)

model.load_state_dict(
    data["model_state"]
)

model.eval()


# --------------------------------------------------
# Load Intent Dataset
# --------------------------------------------------

with open(
    "data/intents.json",
    "r"
) as file:

    intents = json.load(file)


# --------------------------------------------------
# Intro
# --------------------------------------------------

print("\n" + "=" * 60)
print("CHATBOT BASICS PROJECT")
print("=" * 60)

print("\nConversation Pipeline\n")

print("""
User Input
      ↓
Tokenization
      ↓
Bag of Words
      ↓
Neural Network
      ↓
Intent Prediction
      ↓
Response Selection
      ↓
Bot Response
""")

print("=" * 60)

print("\nVocabulary Learned:\n")
print(all_words)

print("\nIntent Classes:\n")
print(tags)

print("\nNeural Network Architecture:\n")
print(model)

print("\nType 'quit' to exit.\n")


# --------------------------------------------------
# Chat Loop
# --------------------------------------------------

while True:

    print("\n" + "=" * 60)

    sentence = input("You: ")

    if sentence.lower() == "quit":

        print("\nGoodbye!\n")

        break

    # --------------------------------------------------
    # Step 1: Tokenization
    # --------------------------------------------------

    tokens = tokenize(sentence)

    print("\n[1] TOKENIZATION")
    print(tokens)

    # --------------------------------------------------
    # Step 2: Bag of Words
    # --------------------------------------------------

    bow = bag_of_words(
        tokens,
        all_words
    )

    print("\n[2] BAG OF WORDS VECTOR")
    print(bow)

    # --------------------------------------------------
    # Step 3: Tensor Conversion
    # --------------------------------------------------

    X = torch.tensor(
        bow,
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Step 4: Neural Network Prediction
    # --------------------------------------------------

    output = model(X)

    probabilities = torch.softmax(
        output,
        dim=0
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=0
    )

    tag = tags[
        predicted.item()
    ]

    # --------------------------------------------------
    # Step 5: Intent Probabilities
    # --------------------------------------------------

    print("\n[3] INTENT PROBABILITIES")

    for intent_name, prob in zip(
        tags,
        probabilities
    ):

        print(
            f"{intent_name:<10} "
            f"-> {prob.item():.4f}"
        )

    # --------------------------------------------------
    # Step 6: Prediction
    # --------------------------------------------------

    print("\n[4] MODEL PREDICTION")

    print(
        f"Predicted Intent : {tag}"
    )

    print(
        f"Confidence       : "
        f"{confidence.item():.2%}"
    )

    # --------------------------------------------------
    # Step 7: Unknown Intent Handling
    # --------------------------------------------------

    if confidence.item() < 0.60:

        print(
            "\nBot: I'm not sure I understand."
        )

        continue

    # --------------------------------------------------
    # Step 8: Response Generation
    # --------------------------------------------------

    response = None

    for intent in intents["intents"]:

        if intent["tag"] == tag:

            response = random.choice(
                intent["responses"]
            )

            break

    print("\n[5] RESPONSE GENERATION")

    print(
        f"Intent '{tag}' matched."
    )

    print(
        f"Selected Response:"
    )

    print(
        f"Bot: {response}"
    )