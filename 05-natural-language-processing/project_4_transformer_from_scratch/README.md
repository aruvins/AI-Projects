# 🧠 Transformer From Scratch

A deep learning project that implements the core components of the Transformer architecture from scratch using PyTorch.

Unlike high-level NLP libraries that hide the underlying mechanics, this project builds the fundamental Transformer components manually, including self-attention, positional encoding, feed-forward networks, residual connections, and layer normalization.

The model is trained on the IMDB movie review dataset to perform sentiment classification while exposing the inner workings of modern language models.

---

# 🎯 Project Goals

This project was created to develop an intuitive understanding of the architecture behind modern AI systems such as:

* GPT
* BERT
* T5
* BART
* Llama

By the end of this project you will understand:

* How Transformers process language
* Why attention replaced recurrent networks
* How self-attention captures context
* Why positional encoding is necessary
* How token embeddings become contextual representations
* How modern Large Language Models are built

---

# 🏗 Project Architecture

```text
Input Text
     ↓
Tokenization
     ↓
Vocabulary Encoding
     ↓
Embedding Layer
     ↓
Positional Encoding
     ↓
Self-Attention
     ↓
Feed Forward Network
     ↓
Classification Head
     ↓
Sentiment Prediction
```

---

# 📂 Project Structure

```text
project_4_transformer_from_scratch/

├── data/
│
├── models/
│   ├── attention.py
│   ├── transformer_block.py
│   └── transformer.py
│
├── utils/
│   ├── imdb_loader.py
│   ├── dataset.py
│   ├── trainer.py
│   └── visualization.py
│
├── outputs/
│   ├── transformer_model.pth
│   ├── attention_map.png
│   ├── training_loss.png
│   ├── validation_accuracy.png
│   └── confusion_matrix.png
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 05-natural-language-processing/project_4_transformer_from_scratch
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate

```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Run the Project
```bash
python main.py
```


If a trained model already exists:

```text
outputs/transformer_model.pth
```

the project automatically loads the saved model instead of retraining.

---

# 📚 Dataset

This project uses the Stanford IMDB Movie Review Dataset.

The dataset contains:

* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment labels

```text
Positive Review → 1
Negative Review → 0
```

Example:

```text
"This movie was fantastic and emotionally engaging."
```

Label:

```text
1
```

---

# 🔤 Tokenization

Neural networks cannot process raw text.

Text must first be converted into tokens.

Example:

```text
I love transformers
```

becomes:

```python
["i", "love", "transformers"]
```

Each token is then mapped into a vocabulary.

```python
{
    "<pad>": 0,
    "<unk>": 1,
    "i": 2,
    "love": 3,
    "transformers": 4
}
```

Result:

```python
[2, 3, 4]
```

---

# 🧮 Word Embeddings

Neural networks cannot learn from integer IDs directly.

Each token is converted into a dense vector.

Example:

```text
love
```

might become:

```python
[0.23, -0.71, 1.54, 0.11]
```

Embeddings allow the model to learn semantic relationships.

Example:

```text
king  - man + woman ≈ queen
```

Words with similar meanings become close together in vector space.

---

# 📍 Positional Encoding

Transformers process every word simultaneously.

Unlike RNNs and LSTMs, they have no inherent notion of order.

Consider:

```text
I love transformers

transformers love I
```

Without positional information these sentences appear identical.

To solve this problem, Transformers add a positional encoding vector to each embedding.

The original Transformer paper introduced sinusoidal positional encodings.

Even dimensions:

```text
sin(position)
```

Odd dimensions:

```text
cos(position)
```

This creates a unique positional signature for every token position.

---

# 🎯 Self-Attention

Self-attention is the core innovation behind Transformers.

Instead of processing words sequentially, every word can examine every other word.

Example:

```text
The dog chased the ball.
```

When processing:

```text
ball
```

the model may attend strongly to:

```text
dog
chased
```

and less strongly to:

```text
the
```

This allows the network to learn context dynamically.

The attention mechanism computes:

Q = Query

K = Key

V = Value

and produces attention weights that determine which words are most important.

---

# ⚡ Scaled Dot Product Attention

The Transformer computes attention scores between all tokens.

The attention operation is:

Attention(Q,K,V)

where:

* Q = Query matrix
* K = Key matrix
* V = Value matrix

The resulting attention matrix determines how strongly each token should focus on every other token.

This enables long-range dependency modeling that was difficult for traditional RNNs.

---

# 🏗 Transformer Block

A Transformer block contains two major components:

## 1. Self-Attention

Learns contextual relationships.

```text
Input
   ↓
Self Attention
```

## 2. Feed Forward Network

Learns higher-level representations.

```text
Attention Output
        ↓
Feed Forward Network
```

---

# 🔄 Residual Connections

Deep neural networks become difficult to train as they grow larger.

Residual connections help preserve information.

Instead of learning:

```text
Output
```

the network learns:

```text
Output + Input
```

Benefits:

* Better gradient flow
* Faster convergence
* Improved stability

This idea was originally popularized by ResNet.

---

# 📏 Layer Normalization

Training deep networks can be unstable.

Layer normalization rescales activations:

```text
Mean = 0
Variance = 1
```

Benefits:

* Faster training
* More stable gradients
* Improved convergence

LayerNorm is applied after both attention and feed-forward layers.

---

# 🎬 Sentiment Classification

After attention processing, token representations are combined into a document representation.

The final classification layer predicts:

```text
Positive
or
Negative
```

using:

```text
Softmax
```

The class with the highest probability becomes the prediction.

---

# 📉 Loss Function

This project uses Cross Entropy Loss.

Cross entropy measures the difference between:

```text
Predicted Probabilities
```

and

```text
Actual Labels
```

Lower loss indicates better predictions.

---

# ⚙️ Optimization

The model is trained using the Adam optimizer.

Adam combines:

* Momentum
* Adaptive Learning Rates

Benefits:

* Faster convergence
* Stable training
* Strong performance on NLP tasks

---

# 📊 Visualizations

The project generates several visualizations.

## Attention Map

Shows how strongly each token attends to every other token.

Example:

```text
dog → ball
```

may receive a strong attention weight.

Output:

```text
outputs/attention_map.png
```

---

## Training Loss Curve

Tracks loss over training epochs.

Output:

```text
outputs/training_loss.png
```

---

## Validation Accuracy Curve

Tracks classification accuracy over time.

Output:

```text
outputs/validation_accuracy.png
```

---

## Confusion Matrix

Visualizes prediction quality.

Output:

```text
outputs/confusion_matrix.png
```

Example:

```text
                 Predicted
              Neg      Pos

Actual Neg     92       8
Actual Pos      6      94
```

---

# 🎓 Key Takeaways

This project demonstrates the fundamental ideas behind modern language models:

* Tokenization
* Embeddings
* Positional Encoding
* Self-Attention
* Transformer Blocks
* Residual Connections
* Layer Normalization
* Classification Heads
* Deep Learning Optimization

These concepts form the foundation of today's state-of-the-art NLP systems, including GPT, BERT, T5, BART, Llama, and many other Large Language Models.
