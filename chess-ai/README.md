# ♟️ Neural Chess AI (V2 - Synthetic Data + Stockfish)

A machine learning chess project that trains a neural network to evaluate chess positions using **synthetically generated board states** labeled by Stockfish.

This version improves significantly over PGN-based training by:
- generating balanced chess positions
- removing dataset download/parsing bottlenecks
- producing faster, more stable training data

---

## 🧠 How It Works

Instead of learning from human games, the AI learns from **random chess positions**:

### Training pipeline
1. Generate a random legal chess position
2. Label it using Stockfish evaluation (centipawns)
3. Convert board into a tensor representation
4. Train a CNN to predict position strength

### Playing pipeline
1. User makes a move in the GUI
2. AI simulates all legal responses
3. Neural network evaluates resulting positions
4. AI selects the highest-scoring move

---

## 📁 Project Structure


chess-ai/
│
├── main.py                     # Entry point (menu system)
├── requirements.txt
├── README.md
│
├── chess_ai/
│   ├── gui.py                  # Pygame chess interface
│   ├── ai.py                  # Move selection logic
│   ├── model.py               # Neural network (CNN)
│   ├── train.py               # Training pipeline
│   ├── synthetic_data.py      # NEW: random position generator
│   ├── stockfish_eval.py      # Stockfish evaluation wrapper
│   └── data_loader.py         # Board encoding utilities
│
└── chess_net.pth              # Trained model (generated after training)


---
# Installation

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```
python-chess
pygame
numpy
torch
tqdm
stockfish
```

---

## ♟️ Install Stockfish

### macOS (Homebrew)

```bash
brew install stockfish
```

### Linux

```bash
sudo apt install stockfish
```

### Windows

Download from:
[https://stockfishchess.org/download/](https://stockfishchess.org/download/)

---

## 🚀 Running the Project

Start the program:

```bash
python main.py
```

You will see:

```
1. Train model
2. Play vs AI
3. Exit
```

---

## 🧠 Training the AI

When you select:

```
1. Train model
```

The system will:

* generate random chess positions
* evaluate them using Stockfish
* build a training dataset in memory
* train a convolutional neural network

---

## ⚡ Why Synthetic Data?

Instead of using chess games (PGNs), this version generates positions directly.

### Benefits:

* no dataset downloads required
* no parsing bottlenecks
* balanced distribution of positions
* better coverage of tactical and endgame scenarios
* faster iteration cycles

---

## 🧠 Model Architecture

A convolutional neural network:

* Input: 12 × 8 × 8 board tensor
* 2 convolution layers
* Fully connected layers
* Output: scalar evaluation (centipawn score)

---

## 📊 Training Pipeline

```
Random Position Generator
        ↓
Stockfish Evaluation (label)
        ↓
Tensor Encoding
        ↓
Neural Network Training
```

---

## ⚠️ Limitations (V2)

This system is still a learning project:

* no search tree (no minimax or MCTS)
* Stockfish used only for labels, not gameplay
* model is purely positional (no planning)
* strength depends on dataset size and Stockfish depth

---

## 🚀 Future Improvements

### 🔥 AI Strength

* Add Monte Carlo Tree Search (MCTS)
* Add policy network (move prediction)
* Add self-play reinforcement learning (AlphaZero-style)

### ⚡ Performance

* cached Stockfish evaluations (SQLite DB)
* multiprocessing dataset generation
* GPU-accelerated training

### 📈 Analysis

* training loss visualization (TensorBoard)
* Elo rating system vs baseline bots
* evaluation benchmarks vs Stockfish

---

## 🎯 Learning Goals

This project teaches:

* how neural networks evaluate structured states
* how synthetic data improves ML pipelines
* how supervised learning works in game AI
* how to build full end-to-end ML systems

---

## 🛠 Tech Stack

* Python 3.9+
* PyTorch
* python-chess
* pygame
* Stockfish
* tqdm

---

## 📄 License

Educational use only.
