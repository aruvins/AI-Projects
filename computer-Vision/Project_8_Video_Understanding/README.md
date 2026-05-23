# Video Transformer — Action Recognition from Scratch

A Video Transformer model built entirely from scratch in PyTorch, with no pretrained weights. It watches a short video clip and predicts what action is happening — for example: *Basketball*, *Archery*, or *Diving*. This project is designed to teach you how modern video AI models actually work, one component at a time.

---

## What does this project do?

Imagine showing a short video clip to a model and having it answer: *"What action is being performed?"* That's called **action recognition** — and it's the core task this model solves.

This project trains a **Video Transformer** on UCF-101, a standard benchmark dataset of 101 human action classes (things like basketball, archery, and diving), containing ~13,000 short clips.

---

## Project files

```
video-transformer/
├── video_transformer.py   ← The model + training loop (main file)
├── download_data.py       ← Downloads & prepares the UCF-101 dataset
├── requirements.txt       ← Python package dependencies
└── README.md              ← This file
```

---

## Quick start (5 minutes, no download required)

If you just want to see the model run right now, use the **synthetic mode**, which generates random fake videos so you can train and test without downloading anything.

```bash
# 1. Clone or copy all project files into a folder, then enter it
cd computer-Vision/Project_8_Video-Understanding

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
pip install -r requirements.txt

# 3. Activate the environment
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# 4. Generate a small fake dataset
python download_synthetic_data.py --output_dir ./data

# 5. Verify the model works (no training, just a shape check)
python video_transformer.py

# 6. Train on the synthetic data
python video_transformer.py --train --data ./data --epochs 10
```

---

## Full setup with real data (UCF-101)

```bash
# Download and preprocess UCF-101 (~6 GB, takes 10–30 min)
https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

# Train (30 epochs is a good starting point)
python video_transformer.py --train --data ./data --epochs 30
```

> **Tip:** If you only want a quick real-data test, limit the number of videos:
> ```bash
> python download_data.py --output_dir ./data --max_videos 500
> ```

---

## How the model works

This is the most important section. Every component below is written out in full in `video_transformer.py` — nothing is hidden inside a library.

### The big picture

A regular image model sees one photo. A video model sees a sequence of frames. The challenge is figuring out **what changes over time** — for example, a person raising their arm versus swinging it.

Our model reads 16 evenly-spaced frames from a clip and learns patterns across both space (where things are) and time (how they move).

---

### Step 1 — Tubelet Embedding

**What it is:** The model can't look at a raw video directly. First, we slice the video into small 3D blocks called **tubelets** — cubes that span a small region of space *and* a couple of frames in time.

**The analogy:** Think of cutting a loaf of bread into uniform cubes. Each cube captures a little slice of motion.

**What happens technically:** Each tubelet is flattened and multiplied by a learned weight matrix to produce a single number-vector called a **token**. This is done in one step using a 3D convolution. A 16-frame, 112×112 video produces **392 tokens**.

```
Input:  (Batch, 3 channels, 16 frames, 112 height, 112 width)
Output: (Batch, 392 tokens, 192 dimensions each)
```

---

### Step 2 — The [CLS] Token

**What it is:** Before feeding the tokens into the Transformer, we prepend one special learnable token called the **[CLS] token** (short for "classification").

**The analogy:** Imagine adding a blank summary card at the front of a stack of index cards. By the end of processing, this card will have collected information from all the others and will be used to make the final prediction.

After the Transformer processes all 393 tokens (1 CLS + 392 video tokens), only the CLS token's output is handed to the classifier.

---

### Step 3 — Positional Embedding

**What it is:** Transformers don't know the order of their inputs — all tokens look the same to them. We fix this by adding a small learnable position vector to each token.

**The analogy:** It's like writing a seat number on each index card. Token 42 always has a small nudge that says "I came from position 42."

This lets the model learn things like "token 0 is the CLS token," "tokens 1–49 are in the first temporal segment," etc.

---

### Step 4 — Multi-Head Self-Attention

This is the **heart of the Transformer**. It's the mechanism that lets every token look at every other token and decide what information is relevant.

**The analogy:** Imagine a meeting room. Every person (token) can ask a question (Query), every person can answer (Key + Value). You find the most relevant people to listen to by matching your question to their answers, then take a weighted average of what they say.

**The math (simplified):**

```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d ) × V
```

- Q (Query): "What am I looking for?"
- K (Key): "What do I offer?"
- V (Value): "What do I actually contain?"

**Multi-head** means we run this process several times in parallel with different learned Q/K/V projections, then combine the results. Each head can focus on different kinds of relationships (e.g. one head for nearby motion, another for long-range context).

---

### Step 5 — Feed-Forward Network (FFN)

After attention, each token passes through a small two-layer network:

```
token → Linear → GELU activation → Linear → token
```

This lets the model apply a non-linear transformation to the information it just gathered. The hidden layer is 4× larger than the input, giving it room to compute.

---

### Step 6 — Transformer Block

Steps 4 and 5 together form one **Transformer block**. The model stacks 4 of these. Each block also has:

- **LayerNorm** before each sub-layer (stabilises training)
- **Residual connections** — the input is added back to the output, so gradients flow easily during backpropagation

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

---

### Step 7 — Classification Head

After 4 Transformer blocks, we extract just the **[CLS] token**, apply a final LayerNorm, and pass it through a single linear layer that produces one score per class:

```
(Batch, 192)  →  Linear  →  (Batch, 101 classes)
```

The class with the highest score is the prediction.

---

### Full architecture diagram

```
Video clip (B, 3, 16, 112, 112)
        │
        ▼
┌─────────────────────┐
│  Tubelet Embedding  │  ← Conv3D slices video into 392 tokens
└─────────────────────┘
        │
        ▼
  Prepend [CLS] token  →  393 tokens total
        │
        ▼
  + Positional Embedding
        │
        ▼
┌──────────────────────────┐
│  Transformer Block × 4   │
│  ┌──────────────────┐    │
│  │  LayerNorm       │    │
│  │  Self-Attention  │    │
│  │  + Residual      │    │
│  └──────────────────┘    │
│  ┌──────────────────┐    │
│  │  LayerNorm       │    │
│  │  FFN             │    │
│  │  + Residual      │    │
│  └──────────────────┘    │
└──────────────────────────┘
        │
  Extract [CLS] token
        │
        ▼
  LayerNorm → Linear
        │
        ▼
  Class logits (B, 101)
```

---

## Training details

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 with cosine decay |
| Weight decay | 0.05 |
| Loss function | Cross-entropy with label smoothing 0.1 |
| Gradient clipping | Max norm 1.0 |
| Augmentation | Random horizontal flip, brightness jitter |

**Label smoothing** means instead of training the model to output a hard 1.0 for the correct class, we aim for 0.9. This prevents overconfidence and improves generalisation.

**Cosine decay** gradually reduces the learning rate from 1e-4 down to ~1e-6 over the training run, similar to slowly turning down the dial as the model gets closer to a good solution.

---

## All training options

```bash
python video_transformer.py --train \
  --data          ./data   \   # path to prepared dataset
  --epochs        30       \   # number of full passes through the data
  --batch_size    8        \   # videos per gradient update (lower = less GPU RAM)
  --lr            1e-4     \   # starting learning rate
  --embed_dim     192      \   # token size (larger = more capacity)
  --depth         4        \   # number of Transformer blocks
  --num_heads     3        \   # attention heads per block
  --num_frames    16       \   # frames sampled from each clip
  --img_size      112      \   # spatial resolution (pixels)
  --dropout       0.1          # regularisation strength
```

---

## Data preparation options

```bash
# Tiny synthetic dataset (no internet, great for testing)
python download_data.py --synthetic

# Full UCF-101 dataset
python download_data.py --output_dir ./data

# Limit to N videos (faster iteration)
python download_data.py --output_dir ./data --max_videos 1000

# Change frame count or resolution
python download_data.py --num_frames 8 --img_size 64
```

---

## Requirements

- Python 3.9 or newer
- ~4 GB RAM (CPU training), 6 GB+ VRAM (GPU training)
- ~8 GB disk space for UCF-101

The model automatically uses a GPU if one is available (CUDA). To check:

```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Concepts to explore next

Once you understand this model, natural next steps are:

- **Divided Space-Time Attention** — attending to space and time separately for efficiency (TimeSformer, 2021)
- **Video Masked Autoencoders** — self-supervised pre-training by masking patches (VideoMAE, 2022)
- **Inflated 3D CNNs (I3D)** — an earlier approach using 3D convolutions throughout
- **Optical flow** — explicitly computing motion between frames as a second input modality

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
→ Make sure your virtual environment is activated: `source .venv/bin/activate`

**`CUDA out of memory`**
→ Reduce `--batch_size` to 4 or 2. The model also trains on CPU.

**`FileNotFoundError: Train manifest not found`**
→ Run `python download_data.py --synthetic` first to create the dataset.

**`unrar: command not found`** during UCF-101 download
→ Install it: `sudo apt install unrar` (Linux) or `brew install unar` (macOS). The script will also fall back to a `.tar.gz` mirror automatically.

---
 
## References
 
### Foundational papers
 
These are the papers this project is directly built on. Reading them in order is the best way to understand the progression of ideas.
 
**Attention Is All You Need (Vaswani et al., 2017)**
The original Transformer paper. Introduces the self-attention mechanism, multi-head attention, and the encoder-decoder architecture. Everything in this project traces back here.
https://arxiv.org/abs/1706.03762
 
**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale — ViT (Dosovitskiy et al., 2020)**
Applies the Transformer directly to images by splitting them into patches. The tubelet embedding and [CLS] token in our model are taken directly from this paper.
https://arxiv.org/abs/2010.11929
 
**Is Space-Time Attention All You Need for Video Understanding? — TimeSformer (Bertasius et al., 2021)**
The first paper to apply ViT-style attention to video. Proposes dividing space and time attention into separate steps for efficiency. Very close in spirit to this project.
https://arxiv.org/abs/2102.05095
 
**ViViT: A Video Vision Transformer (Arnab et al., 2021)**
Google's video transformer. Introduces the tubelet embedding used in this project and benchmarks several attention factorisation strategies on Kinetics.
https://arxiv.org/abs/2103.15691
 
**VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training (Tong et al., 2022)**
Shows how to pre-train video transformers without labels by masking ~90% of patches and reconstructing them. A natural next step after supervised training.
https://arxiv.org/abs/2203.12602
 
---
 
### Dataset
 
**UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild (Soomro et al., 2012)**
The dataset used in this project. 13,320 clips across 101 action categories collected from YouTube.
https://arxiv.org/abs/1212.0402
 
UCF-101 download page: https://www.crcv.ucf.edu/data/UCF101.php
 
---
 
### Background reading (beginner-friendly)
 
These are articles and guides — no maths degree required.
 
**The Illustrated Transformer — Jay Alammar**
The clearest visual explanation of how attention and transformers work. Highly recommended before reading any of the papers above.
http://jalammar.github.io/illustrated-transformer/
 
**The Illustrated Self-Supervised Learning — Jay Alammar**
Explains masked autoencoders and contrastive learning in the same approachable visual style. Useful context for VideoMAE.
https://jalammar.github.io/illustrated-self-supervised-learning/
 
**Attention? Attention! — Lilian Weng**
A thorough written walkthrough of every attention variant — soft, hard, self, multi-head. Good reference to keep open while reading the code.
https://lilianweng.github.io/posts/2018-06-24-attention/
 
**A Recipe for Training Neural Networks — Andrej Karpathy**
Practical advice on debugging training runs, choosing hyperparameters, and understanding loss curves. Directly applicable when training this model.
http://karpathy.github.io/2019/04/25/recipe/
 
---
 
### Video lectures
 
**Stanford CS231n — Convolutional Neural Networks for Visual Recognition**
Covers CNNs, RNNs, and vision fundamentals. Lecture 10 covers video and temporal models specifically.
https://cs231n.stanford.edu/
 
**Andrej Karpathy — Let's build GPT from scratch (YouTube)**
Builds a Transformer from scratch in ~2 hours of live coding. The architecture is the same as our TransformerBlock — just applied to text instead of video.
https://www.youtube.com/watch?v=kCc8FmEb1nY
 
**Yannic Kilcher — ViViT paper walkthrough (YouTube)**
A 30-minute video explanation of the ViViT paper, which is the direct inspiration for this project.
https://www.youtube.com/watch?v=TFwqBMT7Eeg