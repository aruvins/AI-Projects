# 👕 Fashion-MNIST Classifier

A Deep Learning project that classifies clothing images from the Fashion-MNIST dataset using a Multi-Layer Perceptron (MLP) built with PyTorch.

This project introduces core neural network concepts including forward propagation, backpropagation, optimization, validation, regularization, overfitting, and model evaluation.

---

# 📂 Project Structure

```text
project_3_fashion_mnist_classifier/

│
├── data/
│
├── outputs/
│   ├── fashion_classifier.pth
│   ├── training_loss.png
│   ├── validation_accuracy.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── main.py
├── utils/
│   └── model.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. CD into Repository

```bash
cd 03-deep-learning-foundations/project_3_fashion_mnist_classifier

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

# 4. Run The Project

```bash
python main.py

```

---

# 🎯 Project Goal

The goal of this project is to build a neural network capable of recognizing different categories of clothing from grayscale images.

Unlike the classic MNIST handwritten digit dataset, Fashion-MNIST is a more challenging computer vision dataset because many clothing items have similar visual features.

Examples include:

* Shirts vs T-Shirts
* Coats vs Pullovers
* Sneakers vs Ankle Boots

The objective is not only to achieve high accuracy, but also to learn how neural networks generalize to unseen data.

---

# 📚 Concepts Covered

This project introduces several important Deep Learning concepts:

* Neural Networks
* Forward Propagation
* Activation Functions
* Loss Functions
* Backpropagation
* Gradient Descent
* Adam Optimizer
* Training vs Validation vs Test Data
* Overfitting
* Regularization
* Dropout
* Model Evaluation
* Confusion Matrices

---

# 🖼️ About Fashion-MNIST

Fashion-MNIST is a computer vision dataset created by Zalando Research.

The dataset contains:

| Dataset  | Images |
| -------- | ------ |
| Training | 60,000 |
| Test     | 10,000 |

Each image is:

```text
28 × 28 grayscale pixels
```

with values ranging from:

```text
0 → black
255 → white
```

The model must classify each image into one of 10 categories.

| Label | Class       |
| ----- | ----------- |
| 0     | T-Shirt/Top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle Boot  |

---

# 🧠 Neural Networks

A neural network is a mathematical function that learns patterns from data.

Instead of manually defining rules, we allow the model to learn useful relationships directly from examples.

The network learns by adjusting parameters called weights.

A neural network consists of:

```text
Input Layer
    ↓
Hidden Layers
    ↓
Output Layer
```

For Fashion-MNIST:

```text
784 Inputs
    ↓
256 Neurons
    ↓
128 Neurons
    ↓
10 Outputs
```

The input layer contains:

```text
28 × 28 = 784 pixels
```

Each pixel becomes a numerical feature used by the model.

---

# 🔄 Forward Propagation

Forward propagation is the process of generating predictions.

The input image passes through every layer of the network.

Each neuron computes:

```math
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
```

Where:

* x = inputs
* w = weights
* b = bias

The result is passed through an activation function before moving to the next layer.

This process continues until the model produces class scores for all clothing categories.

---

# ⚡ ReLU Activation Function

Neural networks require activation functions to learn complex patterns.

This project uses ReLU:

```math
f(x)=\max(0,x)
```

ReLU outputs:

```text
Positive values → unchanged
Negative values → 0
```

Benefits:

* Simple
* Fast
* Helps deep networks learn efficiently
* Reduces vanishing gradient problems

Without activation functions, neural networks would behave like simple linear models regardless of depth.

---

# 🎯 Output Layer

The final layer contains:

```python
nn.Linear(128, 10)
```

The network produces 10 scores:

```text
[2.1, 0.4, 1.8, 4.5, ...]
```

Each score represents the model's confidence for a clothing category.

The highest score becomes the prediction.

For example:

```text
Class 3 (Dress)
```

would be selected if it has the highest score.

---

# 🏗️ Understanding the Network Architecture

The Fashion-MNIST classifier is built using several different neural network layers. Two of the most important components are:

* `nn.Linear()`
* `nn.ReLU()`

Understanding why these layers are used helps explain how neural networks learn patterns from data.

---

# 🔹 Why Do We Use `nn.Linear()`?

A Linear layer is the most basic building block of a neural network.

In PyTorch:

```python
nn.Linear(input_features, output_features)
```

For example:

```python
nn.Linear(784, 256)
```

means:

```text
784 input features
        ↓
256 output features
```

Each output neuron is connected to every input neuron.

Because every neuron connects to every neuron in the next layer, this is often called a:

```text
Fully Connected Layer
```

or

```text
Dense Layer
```

---

## What Does a Linear Layer Actually Do?

A Linear layer performs a mathematical operation:

```math
y = Wx + b
```

Where:

* `x` = input values
* `W` = learned weights
* `b` = learned bias
* `y` = output values

The weights and biases are the parameters the model learns during training.

You can think of the Linear layer as:

```text
A feature combiner
```

It learns which input features are important and how they should be combined.

For example:

```text
Pixel A → important
Pixel B → less important
Pixel C → very important
```

The network learns these relationships automatically.

---

## Why Is `nn.Linear()` Used In This Project?

Fashion-MNIST images are relatively small:

```text
28 × 28 pixels
```

After flattening:

```text
28 × 28 = 784 features
```

An MLP (Multi-Layer Perceptron) works well because:

* images are small
* architecture is simple
* easy for beginners to understand
* introduces core neural network concepts

The Linear layers allow the network to learn patterns from the image pixels.

---

## Other Types of Layers

Linear layers are not the only option.

### 1. Convolutional Layers

```python
nn.Conv2d()
```

Used in:

* CNNs
* Computer Vision
* Image Classification

Example:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3
)
```

Advantages:

* Detect edges
* Detect textures
* Detect shapes
* Require fewer parameters

CNNs usually outperform MLPs on image tasks.

---

### 2. Recurrent Layers

```python
nn.RNN()
nn.LSTM()
nn.GRU()
```

Used for:

* Text
* Language
* Time Series
* Speech

These layers can remember previous information.

---

### 3. Transformer Layers

```python
nn.TransformerEncoder()
```

Used for:

* Large Language Models
* GPT-style systems
* Modern AI

Transformers learn relationships between all inputs simultaneously.

---

# 🔹 Why Do We Use `nn.ReLU()`?

Without activation functions, neural networks cannot learn complex patterns.

Suppose we had:

```python
nn.Linear(784, 256)
nn.Linear(256, 128)
nn.Linear(128, 10)
```

with no activation functions.

Even though there are multiple layers, the network would behave exactly like:

```python
nn.Linear(784, 10)
```

The extra layers would provide no additional learning power.

Activation functions solve this problem.

---

## What Does ReLU Do?

ReLU stands for:

```text
Rectified Linear Unit
```

It applies:

```math
f(x) = max(0, x)
```

Examples:

```text
Input  → Output

5      → 5
2      → 2
0      → 0
-3     → 0
-10    → 0
```

Negative values become zero.

Positive values remain unchanged.

---

## Why Is ReLU Popular?

ReLU became the standard activation function because it is:

### Fast

Very simple calculation:

```text
max(0, x)
```

This makes training efficient.

---

### Helps Deep Networks Learn

Older activation functions often caused:

```text
Vanishing Gradients
```

where gradients became extremely small.

When gradients become tiny:

```text
Weights stop updating
Learning slows dramatically
```

ReLU greatly reduces this problem.

---

### Produces Sparse Activations

Many neurons become:

```text
0
```

for a given input.

This often improves efficiency and generalization.

---

# Other Activation Functions

ReLU is not the only option.

---

## Sigmoid

```python
nn.Sigmoid()
```

Formula:

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

Output range:

```text
0 → 1
```

Useful for:

* Binary Classification
* Probabilities

Example:

```text
Spam vs Not Spam
Fraud vs Not Fraud
```

Problem:

```text
Vanishing Gradients
```

in deep networks.

---

## Tanh

```python
nn.Tanh()
```

Output range:

```text
-1 → 1
```

Advantages:

* Zero centered
* Stronger gradients than Sigmoid

Still suffers from vanishing gradients.

---

## Leaky ReLU

```python
nn.LeakyReLU()
```

Instead of:

```text
negative values → 0
```

it allows a small negative slope.

Example:

```text
-10 → -0.1
```

Benefits:

* Prevents dead neurons
* Often performs similarly to ReLU

---

## GELU

```python
nn.GELU()
```

Used by:

* Transformers
* BERT
* GPT models

Advantages:

* Smooth activation
* Strong performance in large models

Disadvantages:

* More computationally expensive

---

## ELU

```python
nn.ELU()
```

Another alternative to ReLU.

Benefits:

* Allows negative outputs
* Helps gradient flow

Often used in deeper architectures.

---

# Why ReLU Is The Best Choice Here

For a beginner Fashion-MNIST classifier:

```python
nn.ReLU()
```

is usually the best choice because it is:

* Simple
* Fast
* Easy to understand
* Industry standard
* Works extremely well for MLPs

More advanced projects may replace ReLU with:

```python
nn.LeakyReLU()
nn.GELU()
```

but ReLU remains the most common starting point when learning Deep Learning.

---

# Summary

| Component                 | Purpose                                               |
| ------------------------- | ----------------------------------------------------- |
| `nn.Linear()`             | Learns relationships between features                 |
| `nn.ReLU()`               | Adds non-linearity so complex patterns can be learned |
| `nn.Conv2d()`             | Alternative layer for images                          |
| `nn.LSTM()`               | Alternative layer for sequences                       |
| `nn.TransformerEncoder()` | Alternative layer for modern AI systems               |
| `nn.Sigmoid()`            | Binary classification activation                      |
| `nn.Tanh()`               | Activation with outputs from -1 to 1                  |
| `nn.LeakyReLU()`          | ReLU variant that avoids dead neurons                 |
| `nn.GELU()`               | Modern activation used in Transformers                |

Together, the Linear layers learn useful feature representations while ReLU gives the network the ability to model complex, non-linear relationships in the Fashion-MNIST dataset.

---

# 📉 Loss Function

The loss function measures prediction error.

This project uses:

```python
nn.CrossEntropyLoss()
```

Cross-Entropy compares:

```text
Predicted probabilities
vs
Correct label
```

If the prediction is wrong:

```text
Large Loss
```

If the prediction is correct:

```text
Small Loss
```

The objective of training is to minimize this loss.

---

# 🔙 Backpropagation

Backpropagation is how neural networks learn.

After computing loss:

1. Calculate prediction error
2. Compute gradients
3. Send gradients backward through the network
4. Update weights

The gradient tells the model:

```text
Which direction should each weight move?
```

Weights that contribute to errors are adjusted accordingly.

This process occurs after every training batch.

---

# ⛰️ Gradient Descent

Gradient Descent is the optimization process used to minimize loss.

Imagine standing on a mountain in dense fog.

Your objective is to reach the lowest point.

The gradient tells you:

```text
Which direction is downhill?
```

The optimizer repeatedly updates weights until the loss becomes as small as possible.

---

# 🚀 Adam Optimizer

This project uses:

```python
torch.optim.Adam()
```

Adam stands for:

```text
Adaptive Moment Estimation
```

Adam improves upon standard Gradient Descent by tracking:

### First Moment

The average direction of recent gradients.

```text
Momentum
```

This helps the model move consistently toward better solutions.

### Second Moment

The average magnitude of recent gradients.

This allows Adam to automatically adjust learning rates for individual parameters.

Benefits:

* Fast convergence
* Stable training
* Minimal tuning required
* Excellent default optimizer

Because of these advantages, Adam is one of the most widely used optimizers in modern Deep Learning.

---
# 📦 Mini-Batch Training

Training uses:

```python
batch_size = 64
```

Instead of processing one image at a time, the model processes:

```text
64 images simultaneously
```

Mini-batch training is one of the most important ideas in Deep Learning because it balances learning quality and computational efficiency.

---

# Why Not Train One Image At A Time?

Suppose we have:

```text
60,000 training images
```

One approach would be:

```text
Image 1 → Update weights
Image 2 → Update weights
Image 3 → Update weights
...
```

This is called:

```text
Stochastic Gradient Descent (SGD)
```

where the model updates its weights after every individual training example.

While this works, it creates a problem.

Each image contains different information.

For example:

```text
Image 1 = Sneaker
Image 2 = Shirt
Image 3 = Coat
```

The gradient calculated from a single image can be noisy.

As a result, weight updates may look like:

```text
←
↗
↓
→
↖
```

instead of moving consistently toward a better solution.

Training becomes unstable and inefficient.

---

# Why Not Use The Entire Dataset?

At the opposite extreme, we could process all training examples at once:

```text
60,000 images
        ↓
One giant forward pass
        ↓
One giant gradient update
```

This is called:

```text
Batch Gradient Descent
```

Advantages:

* Very stable gradients
* Uses information from the entire dataset

Problems:

* Extremely memory intensive
* Very slow updates
* Doesn't scale to large datasets

Imagine training GPT-style models with:

```text
Millions or billions of examples
```

Processing everything simultaneously would be impossible.

---

# Mini-Batch Gradient Descent

Modern Deep Learning uses a compromise:

```text
Mini-Batch Gradient Descent
```

Instead of:

```text
1 image
```

or

```text
60,000 images
```

we use:

```text
64 images
```

at a time.

Example:

```text
Batch 1 → Images 1-64
Batch 2 → Images 65-128
Batch 3 → Images 129-192
...
```

Each mini-batch produces:

* one forward pass
* one loss calculation
* one backward pass
* one optimizer update

---

# What Happens Inside A Batch?

Suppose:

```python
batch_size = 64
```

Each Fashion-MNIST image contains:

```text
28 × 28 = 784 pixels
```

When a batch is loaded:

```python
images.shape
```

becomes:

```text
(64, 1, 28, 28)
```

meaning:

```text
64 images
1 channel
28 rows
28 columns
```

The model processes all 64 images simultaneously.

---

# Forward Pass For A Batch

The batch enters the network:

```text
(64, 784)
        ↓
Linear Layer
        ↓
(64, 256)
        ↓
ReLU
        ↓
(64, 256)
        ↓
Linear Layer
        ↓
(64, 128)
        ↓
ReLU
        ↓
(64, 128)
        ↓
Output Layer
        ↓
(64, 10)
```

Notice:

```text
64 examples move through the network together
```

PyTorch performs these operations using highly optimized matrix multiplication.

This is one reason GPUs are so effective.

---

# Computing The Loss

After the forward pass, the model produces:

```text
64 predictions
```

For example:

```text
Image 1 → Sneaker
Image 2 → Shirt
Image 3 → Coat
...
Image 64 → Bag
```

The loss function compares:

```text
Predicted labels
vs
True labels
```

for all 64 images.

CrossEntropyLoss computes:

```text
Average error across the batch
```

instead of using only a single example.

This produces a more reliable estimate of model performance.

---

# Backpropagation On A Batch

Once the loss is computed:

```python
loss.backward()
```

PyTorch calculates gradients for every parameter.

Importantly:

```text
The gradients are based on all 64 images
```

not just one.

This means the gradient represents the average learning signal from multiple examples.

Instead of learning from:

```text
One shirt
```

the model learns from:

```text
Multiple shirts
Multiple shoes
Multiple coats
Multiple bags
```

at the same time.

This produces more stable updates.

---

# Weight Update

After gradients are computed:

```python
optimizer.step()
```

updates every weight.

Conceptually:

```text
Current Weights
       ↓
Compute Gradients
       ↓
Update Weights
       ↓
New Weights
```

One batch produces one update.

---

# 🔄 How Epochs And Batches Work Together

One of the most common sources of confusion when learning Deep Learning is understanding the relationship between:

* Dataset
* Batch
* Epoch

A useful way to think about it is:

```text
Dataset → Split into Batches → Processed During an Epoch
```

---

# Step 1: Start With The Dataset

Fashion-MNIST contains:

```text
60,000 training images
```

These images represent the entire training dataset.

Before training begins, PyTorch shuffles the dataset:

```python
DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
```

Shuffling helps prevent the model from learning patterns based on data order.

---

# Step 2: Create Mini-Batches

Using:

```python
batch_size = 64
```

PyTorch divides the dataset into groups of 64 images.

Conceptually:

```text
Batch 1   → Images 1-64
Batch 2   → Images 65-128
Batch 3   → Images 129-192
...
Batch 938 → Remaining Images
```

For Fashion-MNIST:

```python
60000 / 64 ≈ 938
```

So one complete pass through the dataset consists of approximately:

```text
938 batches
```

---

# Step 3: Train On One Batch

The model begins with the first batch.

```text
Batch 1
(64 images)
```

The following steps occur:

### Forward Pass

The images pass through the network.

```text
Images
    ↓
Predictions
```

---

### Loss Calculation

Predictions are compared to the correct labels.

```text
Predictions
      vs
True Labels
```

This produces a loss value.

---

### Backpropagation

The model calculates gradients.

```text
Loss
   ↓
Gradients
```

The gradients indicate how each weight contributed to the error.

---

### Weight Update

The optimizer updates the weights.

```text
Old Weights
      ↓
Gradient Update
      ↓
New Weights
```

At this point:

```text
Batch 1 is finished
```

and the model moves to Batch 2.

---

# Step 4: Repeat For Every Batch

The same process repeats:

```text
Batch 1 → Update Weights
Batch 2 → Update Weights
Batch 3 → Update Weights
...
Batch 938 → Update Weights
```

Notice something important:

```text
The model learns after every batch.
```

It does **not** wait until the end of the epoch.

Every batch produces a new version of the model with slightly improved weights.

---

# Step 5: Complete One Epoch

After all batches have been processed:

```text
Batch 1
Batch 2
Batch 3
...
Batch 938
```

the model has now seen every image in the training dataset once.

This is called:

```text
1 Epoch
```

An epoch simply means:

```text
One complete pass through the entire training dataset
```

---

# Visualizing An Epoch

Suppose we have:

```text
60,000 images
Batch Size = 64
```

One epoch looks like:

```text
Dataset
│
├── Batch 1
├── Batch 2
├── Batch 3
├── ...
├── Batch 937
└── Batch 938

↓
Epoch Complete
```

After the final batch finishes:

```text
Epoch 1 Complete
```

---

# What Happens During Multiple Epochs?

Suppose:

```python
epochs = 10
```

Training now looks like:

```text
Epoch 1
 ├─ Batch 1
 ├─ Batch 2
 ├─ ...
 └─ Batch 938

Epoch 2
 ├─ Batch 1
 ├─ Batch 2
 ├─ ...
 └─ Batch 938

...

Epoch 10
 ├─ Batch 1
 ├─ Batch 2
 ├─ ...
 └─ Batch 938
```

Each epoch gives the model another opportunity to refine its weights.

The model gradually improves because it repeatedly sees the training examples and adjusts its parameters.

---

# Why Multiple Epochs Are Necessary

Imagine studying for an exam.

Looking at the material once:

```text
1 Epoch
```

might help a little.

Reviewing the material multiple times:

```text
5 Epochs
10 Epochs
20 Epochs
```

usually leads to a deeper understanding.

Neural networks learn in a similar way.

Each epoch reinforces useful patterns and gradually reduces prediction errors.

---

# How Many Weight Updates Occur?

With:

```text
60,000 images
Batch Size = 64
```

there are approximately:

```text
938 batches per epoch
```

Since each batch produces one optimizer update:

```text
938 updates per epoch
```

If training runs for:

```python
epochs = 10
```

then:

```text
938 × 10

=

9,380 weight updates
```

occur during training.

This means the model's weights are adjusted thousands of times before training finishes.

---

# Key Insight

A common beginner misconception is:

```text
1 Epoch = 1 Weight Update
```

This is incorrect.

The correct relationship is:

```text
Dataset
    ↓
Split into Batches
    ↓
Each Batch Updates Weights
    ↓
All Batches Processed
    ↓
Epoch Complete
```

Therefore:

```text
Many Batch Updates
        ↓
Create
        ↓
One Epoch
```

An epoch is simply a measurement of progress through the dataset, while the actual learning happens continuously after every mini-batch.

---

# Why GPUs Love Mini-Batches

GPUs are designed for parallel computation.

Instead of computing:

```text
Image 1
then
Image 2
then
Image 3
```

GPUs can compute:

```text
Images 1-64
```

simultaneously.

Matrix operations become:

```text
Much Faster
Much More Efficient
```

This is why increasing batch size often improves GPU utilization.

---

# Choosing A Batch Size

Common choices include:

```python
32
64
128
256
512
```

There is no universally perfect batch size.

Smaller batches:

```text
More noise
More updates
Less memory usage
```

Larger batches:

```text
Smoother gradients
Fewer updates
More memory usage
```

A batch size of:

```python
64
```

is often a great starting point because it balances:

* speed
* memory consumption
* training stability

---

# Real-World Analogy

Imagine trying to determine whether a restaurant is good.

### One Customer

You ask:

```text
1 person
```

Their opinion might be unusual.

---

### Every Customer

You ask:

```text
100,000 people
```

This would be accurate but extremely slow.

---

### Mini-Batch

You ask:

```text
64 people
```

You receive a reasonably accurate estimate while still getting feedback quickly.

Mini-batch training works the same way.

The model updates its weights using feedback from a representative sample rather than from a single example or the entire dataset.

---

# Why Mini-Batching Is The Standard

Modern Deep Learning almost always uses mini-batches because they provide:

* Faster training
* Better GPU utilization
* More stable gradients
* Lower memory requirements
* Better scalability to large datasets

Without mini-batching, training modern neural networks would be dramatically slower and far less practical.

For this reason, mini-batch gradient descent has become the standard training strategy used in virtually every modern Deep Learning system, from simple Fashion-MNIST classifiers to large language models containing billions of parameters.

---

# 🎲 Train / Validation / Test Split

The dataset is divided into multiple portions.

### Training Set

Used to learn model parameters.

```text
80%
```

### Validation Set

Used to monitor generalization.

```text
20%
```

The validation set helps identify overfitting during training.

### Test Set

Used only after training is complete.

The test set provides an unbiased estimate of real-world performance.

---

# 🎲 Understanding Training, Validation, and Test Sets

One of the most important ideas in Machine Learning is separating data into different datasets.

A common beginner question is:

```text
Why do we need both a Validation Set and a Test Set?
```

At first glance they seem identical because:

```text
Neither dataset is used for learning weights.
```

However, they serve very different purposes.

---

# The Core Goal

The purpose of Machine Learning is not:

```text
Memorizing training data
```

The purpose is:

```text
Generalizing to new data
```

In the real world, the model will encounter images it has never seen before.

We therefore need a way to estimate how well the model performs on unseen examples.

This is where the validation and test sets become important.

---

# The Training Set

The training set is the data used for learning.

For every batch:

```text
Training Images
       ↓
Predictions
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Weight Updates
```

The model directly learns from this data.

In your project:

```text
48,000 images
```

are used for training.

---

# The Validation Set

The validation set acts like a practice exam.

The model does not learn from validation data.

Instead, validation data is used to answer:

```text
"How well is the model generalizing?"
```

After each epoch:

```text
Training Complete
       ↓
Run Validation Set
       ↓
Measure Accuracy
```

No gradients are calculated.

No weights are updated.

The validation set is only used for evaluation.

---

# Why Evaluate After Every Epoch?

Suppose we train:

```python
epochs = 20
```

After every epoch we calculate:

```text
Validation Accuracy
```

For example:

```text
Epoch 1  → 75%
Epoch 2  → 81%
Epoch 3  → 85%
Epoch 4  → 87%
Epoch 5  → 88%
Epoch 6  → 89%
Epoch 7  → 89%
Epoch 8  → 88%
Epoch 9  → 87%
```

Notice what happened:

```text
Validation Accuracy increased
then
started decreasing
```

This is often the first sign of overfitting.

The model is becoming better at memorizing training examples but worse at generalizing.

Without a validation set, you would never notice this happening.

---

# Detecting Overfitting

A common training pattern looks like:

```text
Training Accuracy      Validation Accuracy

90%                    85%
95%                    87%
98%                    86%
99%                    84%
```

The training accuracy continues improving.

The validation accuracy begins decreasing.

This tells us:

```text
The model is memorizing
instead of generalizing.
```

The validation set acts as an early warning system.

---

# Hyperparameter Tuning

The validation set is also used to make decisions about the model.

For example:

```python
batch_size = 32
```

might achieve:

```text
88% Validation Accuracy
```

while:

```python
batch_size = 64
```

achieves:

```text
89% Validation Accuracy
```

We would choose:

```text
Batch Size = 64
```

because it performs better on validation data.

The same applies to:

* Learning Rate
* Number of Layers
* Number of Neurons
* Optimizer Choice
* Dropout Rate
* Number of Epochs

The validation set helps us decide which model configuration is best.

---

# Why Not Use The Test Set For This?

This is where many beginners make a mistake.

Suppose you repeatedly evaluate on the test set:

```text
Try Model A
Check Test Accuracy

Try Model B
Check Test Accuracy

Try Model C
Check Test Accuracy
```

Eventually you begin making decisions based on the test set.

At that point:

```text
The test set is no longer truly unseen.
```

You have indirectly optimized your model for those specific test examples.

This creates an overly optimistic estimate of performance.

---

# The Test Set

The test set acts like the final exam.

It should remain untouched during training.

The workflow should be:

```text
Train Model
      ↓
Evaluate Validation Set
      ↓
Adjust Hyperparameters
      ↓
Train Again
      ↓
Choose Best Model
      ↓
Evaluate Test Set ONCE
```

The test set provides the most honest estimate of how the model will perform in the real world.

---

# School Exam Analogy

A useful analogy is studying for a course.

### Training Set

Homework assignments.

```text
You learn from these.
```

---

### Validation Set

Practice exams.

```text
You check your progress.
You identify weaknesses.
You adjust your studying.
```

---

### Test Set

Final exam.

```text
Taken only after preparation is complete.
```

You would never want to study using the final exam answers.

Similarly:

```text
You should not use the test set
to make training decisions.
```

---

# What Happens If We Skip The Validation Set?

Suppose we train for:

```python
epochs = 50
```

Without validation data, we only see:

```text
Training Loss ↓
Training Accuracy ↑
```

Everything appears to be improving.

However, the model may already be overfitting.

The validation set reveals what is happening on unseen data while training is still in progress.

Without it, determining the best stopping point becomes difficult.

---

# Early Stopping

Many real-world systems use validation performance to decide when to stop training.

Example:

```text
Epoch 1 → 82%
Epoch 2 → 86%
Epoch 3 → 88%
Epoch 4 → 89%
Epoch 5 → 89%
Epoch 6 → 88%
Epoch 7 → 87%
```

The best validation performance occurred at:

```text
Epoch 4
```

Training beyond that point may actually hurt generalization.

This technique is called:

```text
Early Stopping
```

and relies entirely on the validation set.

---

# How Your Project Uses Validation Data

In your Fashion-MNIST classifier:

```python
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
```

This creates:

```text
48,000 Training Images
12,000 Validation Images
```

During training:

```text
Train on 48,000 Images
        ↓
Evaluate on 12,000 Images
        ↓
Compute Validation Accuracy
```

The validation images never participate in weight updates.

They are only used to measure performance.

---

# Validation Set vs Test Set

| Feature                       | Validation Set        | Test Set         |
| ----------------------------- | --------------------- | ---------------- |
| Used During Training?         | Yes (evaluation only) | No               |
| Updates Weights?              | No                    | No               |
| Used To Tune Hyperparameters? | Yes                   | No               |
| Used To Detect Overfitting?   | Yes                   | No               |
| Evaluated Multiple Times?     | Yes                   | Ideally once     |
| Represents Final Performance? | Not exactly           | Yes              |
| Purpose                       | Model Development     | Final Evaluation |

---

# Key Takeaway

Although both datasets contain unseen examples, they serve different roles.

The validation set answers:

```text
How is training going?
```

The test set answers:

```text
How well does the final model actually perform?
```

Think of the validation set as your practice exam and the test set as your final exam.

You are allowed to learn from the practice exam results.

You should not learn from the final exam results.

That separation is what gives the test set its value as an unbiased measure of real-world model performance.

---

# ⚠️ Overfitting

Overfitting occurs when a model memorizes training examples instead of learning general patterns.

Symptoms:

```text
Training Accuracy ↑
Validation Accuracy ↓
```

An overfit model performs well on training data but poorly on unseen data.

The goal of Deep Learning is not memorization.

The goal is generalization.

---

# 🛡️ Regularization

Regularization techniques help prevent overfitting.

This project uses Dropout regularization.

Regularization encourages the model to learn robust patterns that transfer to new data.

---

# 🎲 Dropout: Preventing Neural Networks From Memorizing

Dropout is one of the most widely used regularization techniques in Deep Learning.

In this project, the model uses:

```python
nn.Dropout(0.3)
```

which means:

```text
30% of neurons are randomly disabled
during each training pass
```

At first glance, this may seem strange.

Why would we intentionally remove part of the network while training?

The answer lies in one of the biggest challenges in Machine Learning:

```text
Overfitting
```

---

# The Problem: Overfitting

Suppose we train a neural network long enough.

Eventually it may begin to memorize the training data.

Instead of learning general patterns like:

```text
Shirts tend to have sleeves
Sneakers have distinct soles
Bags have handles
```

it may start memorizing very specific details from individual images.

This often leads to:

```text
Training Accuracy ↑
Validation Accuracy ↓
```

The model becomes extremely good at recognizing examples it has already seen but performs poorly on new examples.

This is called:

```text
Overfitting
```

Dropout helps combat this problem.

---

# How Dropout Works

Consider a hidden layer:

```text
Input
  ↓
[ N1 N2 N3 N4 N5 ]
  ↓
Output
```

Normally all neurons participate in making predictions.

With:

```python
nn.Dropout(0.3)
```

PyTorch randomly disables 30% of the neurons during training.

A forward pass might look like:

```text
Input
  ↓
[ N1 N2 X N4 X ]
  ↓
Output
```

where:

```text
X = dropped neuron
```

On the next batch:

```text
Input
  ↓
[ X N2 N3 X N5 ]
  ↓
Output
```

Different neurons are removed every time.

The dropout pattern constantly changes.

---

# What Does "Dropped" Mean?

A dropped neuron does not participate in the forward pass.

Its output becomes:

```text
0
```

Example:

```text
Before Dropout

[2.3, 1.1, 4.5, 3.2]

After Dropout

[2.3, 0.0, 4.5, 0.0]
```

The neuron is temporarily ignored.

It contributes nothing to the prediction for that batch.

---

# Why This Helps

Imagine a group project.

Suppose one student does all the work.

The other students become dependent on that person.

If that student is absent:

```text
The entire group struggles.
```

Neural networks can develop similar behavior.

Some neurons become extremely important.

Other neurons learn very little.

The network starts relying too heavily on specific pathways.

Dropout prevents this.

Because neurons are randomly removed:

```text
Any neuron might disappear
at any time.
```

The network is forced to distribute knowledge across many neurons.

No single neuron can become indispensable.

---

# Learning Redundant Representations

Without dropout:

```text
Neuron A learns:
"This looks like a sneaker."
```

Other neurons may simply rely on Neuron A.

The network becomes fragile.

With dropout:

```text
Neuron A might disappear.
```

The model must ensure that:

```text
Neuron B
Neuron C
Neuron D
```

also learn useful information.

The network develops multiple ways of recognizing the same concept.

This makes the model more robust.

---

# What Happens During Every Batch?

Suppose:

```python
batch_size = 64
```

During Batch 1:

```text
Neuron 3 dropped
Neuron 8 dropped
Neuron 15 dropped
```

During Batch 2:

```text
Neuron 2 dropped
Neuron 9 dropped
Neuron 12 dropped
```

During Batch 3:

```text
Neuron 1 dropped
Neuron 7 dropped
Neuron 18 dropped
```

The dropout mask changes continuously.

Every batch trains a slightly different version of the network.

---

# A Network Of Networks

One way to think about dropout is:

```text
Training many neural networks
at the same time
```

Imagine a network with:

```text
100 neurons
```

Every batch uses a different subset:

```text
Network Version A
Network Version B
Network Version C
...
```

Instead of training one model, dropout effectively trains many slightly different models that share parameters.

At inference time, all of those learned representations are combined.

This often improves generalization.

---

# Why Not Drop Too Many Neurons?

Suppose:

```python
nn.Dropout(0.9)
```

Now:

```text
90% of neurons disappear
```

Only:

```text
10%
```

remain active.

The network would struggle to learn.

Too much information is removed.

Typical values are:

```python
0.1
0.2
0.3
0.5
```

Common choices:

| Dropout Rate | Meaning  |
| ------------ | -------- |
| 0.1          | Drop 10% |
| 0.2          | Drop 20% |
| 0.3          | Drop 30% |
| 0.5          | Drop 50% |

For Fashion-MNIST:

```python
nn.Dropout(0.3)
```

is a reasonable balance between:

* learning capacity
* regularization strength

---

# What Happens During Evaluation?

One of the most important details about dropout is:

```text
Dropout only works during training.
```

During evaluation:

```python
model.eval()
```

dropout is automatically disabled.

Every neuron becomes active again.

Example:

```text
Training:

[N1 N2 X N4 X]

Evaluation:

[N1 N2 N3 N4 N5]
```

This ensures predictions are:

```text
Stable
Deterministic
Consistent
```

If dropout remained active during inference:

```text
The same image could produce
different predictions every time.
```

That would be undesirable.

---

# Why Outputs Don't Become Too Large

You might wonder:

```text
If neurons are randomly removed,
won't activations become smaller?
```

PyTorch handles this automatically.

When using:

```python
nn.Dropout(0.3)
```

the remaining neurons are scaled during training.

This keeps the expected activation magnitude approximately the same.

As a result:

```text
Training
and
Evaluation
```

remain consistent.

---

# Dropout vs No Dropout

Without Dropout:

```text
Training Accuracy: 99%
Validation Accuracy: 85%
```

The model memorizes training examples.

---

With Dropout:

```text
Training Accuracy: 95%
Validation Accuracy: 89%
```

Training accuracy may be slightly lower.

However:

```text
Generalization improves.
```

This is usually what we care about.

A model that performs well on new data is more valuable than one that simply memorizes the training set.

---

# Why Dropout Works So Well

Dropout introduces controlled randomness into training.

This randomness forces the network to:

* Learn more robust features
* Avoid memorizing training examples
* Reduce dependency on individual neurons
* Improve generalization to unseen data

In many ways, dropout acts like a stress test.

The model learns to perform well even when parts of the network are temporarily unavailable.

As a result, the final model is often more reliable and less prone to overfitting.

---

# Key Takeaway

Dropout is a regularization technique that randomly disables neurons during training.

```python
nn.Dropout(0.3)
```

means:

```text
30% of neurons are randomly removed
for each training batch.
```

This prevents the network from becoming overly dependent on specific neurons and encourages it to learn more distributed and generalizable representations.

Although dropout may slightly reduce training accuracy, it often improves validation and test performance by reducing overfitting and helping the model generalize better to unseen data.

---

# 🛡️ Other Types of Regularization

Dropout is one of the most popular regularization techniques, but it is far from the only one.

The goal of **regularization** is always the same:

```text
Reduce Overfitting
Improve Generalization
```

Regularization encourages a model to learn meaningful patterns rather than memorizing training examples.

Think of regularization as adding constraints that prevent a neural network from becoming overly specialized to the training data.

---

# Why Regularization Is Necessary

Suppose we build a very large neural network:

```text
784 Inputs
    ↓
4096 Neurons
    ↓
4096 Neurons
    ↓
4096 Neurons
    ↓
10 Outputs
```

This model contains millions of parameters.

If the dataset is relatively small, the network may simply memorize:

```text
Image #1
Image #2
Image #3
...
```

instead of learning general concepts such as:

```text
What makes a sneaker a sneaker?
What makes a coat a coat?
```

Regularization helps prevent this memorization.

---

# 1. Dropout

This project uses:

```python
nn.Dropout(0.3)
```

During training:

```text
30% of neurons are randomly disabled
```

Benefits:

* Reduces neuron co-dependency
* Improves generalization
* Acts like training many networks simultaneously

Dropout is one of the most commonly used regularization methods in neural networks.

---

# 2. L2 Regularization (Weight Decay)

One of the most widely used regularization methods is:

```text
L2 Regularization
```

also called:

```text
Weight Decay
```

Instead of only minimizing prediction error:

```text
Loss
```

the optimizer also penalizes large weights.

Conceptually:

```text
Total Loss

=

Prediction Loss

+

Penalty For Large Weights
```

Large weights often indicate:

```text
Memorization
```

By encouraging smaller weights, the model tends to learn smoother and more general patterns.

---

## How To Use L2 Regularization

In PyTorch:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

The parameter:

```python
weight_decay=1e-4
```

adds L2 regularization.

---

## Why It Works

Suppose a neuron learns:

```text
Weight = 1000
```

This neuron now dominates the network.

L2 regularization discourages this by making large weights expensive.

Instead the network may learn:

```text
Weight = 2.5
Weight = 3.1
Weight = 1.8
```

distributed across multiple neurons.

This often improves generalization.

---

# 3. L1 Regularization

Another technique is:

```text
L1 Regularization
```

L1 also penalizes large weights, but differently.

Unlike L2:

```text
L2 → Makes weights smaller
```

L1 tends to:

```text
Push many weights to exactly zero
```

This creates sparse models.

Example:

```text
Before:

[2.1, 1.7, 0.9, 3.2, 1.4]

After:

[2.1, 0, 0, 3.2, 0]
```

Many features become completely unused.

---

## Benefits Of L1

L1 can:

* Simplify models
* Reduce noise
* Improve interpretability

However, L2 is generally more common in Deep Learning.

---

# 4. Early Stopping

One of the simplest and most effective regularization methods is:

```text
Early Stopping
```

Suppose validation accuracy looks like:

```text
Epoch 1 → 80%
Epoch 2 → 85%
Epoch 3 → 88%
Epoch 4 → 89%
Epoch 5 → 89%
Epoch 6 → 88%
Epoch 7 → 87%
```

The model begins overfitting after:

```text
Epoch 4
```

Instead of continuing training:

```text
Stop Training Early
```

and keep the best model.

---

## Why It Works

Overfitting often happens later in training.

Early stopping prevents the network from memorizing too much.

Many production systems use early stopping automatically.

---

# 5. Data Augmentation

Data augmentation creates additional training examples by modifying existing images.

Examples:

```text
Rotate
Flip
Crop
Zoom
Shift
```

Original:

```text
Sneaker
```

Augmented:

```text
Rotated Sneaker
Shifted Sneaker
Zoomed Sneaker
```

The model effectively sees more data without collecting new images.

---

## Why It Works

The network learns:

```text
A sneaker is still a sneaker
even if its position changes.
```

This improves robustness and reduces overfitting.

Data augmentation is especially important in computer vision.

---

# 6. Batch Normalization

Batch Normalization (often called **BatchNorm**) is a technique that makes neural network training faster, more stable, and sometimes more accurate.

In PyTorch, Batch Normalization layers look like:

```python
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
```

For a fully connected network like this Fashion-MNIST classifier, you would typically use:

```python
nn.Linear(784, 256),
nn.BatchNorm1d(256),
nn.ReLU(),
```

The BatchNorm layer is placed between the Linear layer and the activation function.

---

# Why Do We Need Batch Normalization?

As training progresses, the outputs of each layer can change significantly as the weights are updated.

For example:

```text
Epoch 1

Neuron Output:
[0.2, 0.5, 0.7]
```

Later in training:

```text
Epoch 10

Neuron Output:
[25.4, 48.2, 61.7]
```

Or:

```text
Epoch 20

Neuron Output:
[-100, 250, 500]
```

These constantly changing activation distributions make learning more difficult because every layer must continuously adapt to shifting inputs.

Batch Normalization helps by keeping activations on a more consistent scale throughout training.

This makes optimization easier and allows the network to learn more efficiently.

---

# How Batch Normalization Works

Suppose a mini-batch contains:

```text
[2, 4, 6, 8]
```

BatchNorm first computes:

```text
Mean = 5
Standard Deviation ≈ 2.24
```

It then transforms the values so they have:

```text
Mean = 0
Standard Deviation = 1
```

The normalized outputs become approximately:

```text
[-1.34, -0.45, 0.45, 1.34]
```

As a result, neuron activations remain in a predictable range regardless of how weights change during training.

This leads to:

* Faster convergence
* More stable gradients
* Higher learning rates
* Less sensitivity to initialization

BatchNorm also introduces a small amount of noise because every mini-batch has slightly different statistics. This noise acts as a mild form of regularization and can help reduce overfitting.

---

# BatchNorm vs Dropout

Although both techniques can improve generalization, they solve different problems.

### Dropout

```python
nn.Dropout(0.3)
```

Purpose:

```text
Reduce overfitting
```

Method:

```text
Randomly disable neurons
```

---

### Batch Normalization

```python
nn.BatchNorm1d(256)
```

Purpose:

```text
Stabilize and accelerate training
```

Method:

```text
Normalize activations within each batch
```

---

In modern neural networks, BatchNorm and Dropout are often used together because they address different aspects of training.

A common architecture looks like:

```python
nn.Linear(784, 256),
nn.BatchNorm1d(256),
nn.ReLU(),
nn.Dropout(0.3)
```

where:

```text
Linear Layer     → learns features
BatchNorm        → stabilizes activations
ReLU             → introduces non-linearity
Dropout          → reduces overfitting
```

Together, these components help neural networks train faster while improving their ability to generalize to unseen data.

---

# 7. Reducing Model Size

Sometimes the simplest solution is:

```text
Use a Smaller Network
```

Example:

```text
Large Network

784 → 2048 → 1024 → 512 → 10
```

versus:

```text
Smaller Network

784 → 256 → 128 → 10
```

Smaller models have:

```text
Fewer Parameters
Less Capacity To Memorize
```

This naturally reduces overfitting.

---

# 8. Collecting More Data

The most powerful regularization technique is often:

```text
More Data
```

If a model sees:

```text
1,000 examples
```

it may memorize.

If it sees:

```text
1,000,000 examples
```

memorization becomes much harder.

Large datasets encourage models to learn true underlying patterns.

This is one reason modern AI systems are trained on enormous amounts of data.

---

# Which Regularization Methods Are Most Common?

For modern neural networks, the most common techniques are:

### Computer Vision

* Data Augmentation
* Weight Decay (L2)
* Dropout
* Early Stopping

### NLP and Transformers

* Weight Decay (L2)
* Dropout
* Large Datasets
* Early Stopping

### General Deep Learning

* Weight Decay (L2)
* Dropout
* Batch Normalization
* Early Stopping

---

# Regularization In This Project

Your Fashion-MNIST classifier currently uses:

```python
nn.Dropout(0.3)
```

A common improvement would be:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

This combines:

```text
Dropout
+
L2 Regularization
```

which is a very common setup in real-world neural networks.

---

# Summary

| Technique           | How It Helps                                     |
| ------------------- | ------------------------------------------------ |
| Dropout             | Randomly disables neurons                        |
| L2 (Weight Decay)   | Penalizes large weights                          |
| L1 Regularization   | Pushes weights toward zero                       |
| Early Stopping      | Stops training before overfitting                |
| Data Augmentation   | Creates more training examples                   |
| Batch Normalization | Stabilizes training and adds mild regularization |
| Smaller Models      | Reduces memorization capacity                    |
| More Data           | Forces learning of general patterns              |

All regularization techniques aim to achieve the same goal:

```text
Train a model that performs well
on data it has never seen before.
```

Because in Machine Learning, a model's true value is measured not by how well it remembers the training data, but by how well it generalizes to new examples.

---

# 📈 Training Loss Curve

The training loss visualization shows whether the model is learning.

A healthy curve usually looks like:

```text
High Loss
     ↓
Lower Loss
```

Loss should generally decrease over time.

---

# 📊 Confusion Matrix

A confusion matrix shows which classes are confused by the model.

Example:

```text
Actual Coat
Predicted Pullover
```

This helps identify weaknesses in classification performance.

Fashion-MNIST often shows confusion between:

* Shirts
* Pullovers
* Coats

because these categories share similar visual characteristics.

---

# 🖼️ Sample Predictions

The project generates prediction examples showing:

```text
Input Image
Predicted Class
```

These visualizations help verify that the model is making sensible predictions.

---

# 📈 Expected Results

Typical performance for this architecture:

```text
Validation Accuracy: 87% - 90%
Test Accuracy: 86% - 89%
```

Performance may vary slightly due to random initialization and train-validation splits.

---

# Key Takeaways

By completing this project, you will understand:

* How neural networks learn
* How forward propagation works
* How backpropagation updates weights
* Why loss functions matter
* How Adam optimization works
* Why validation sets are important
* What overfitting looks like
* How Dropout improves generalization
* How to evaluate classification models
* How Deep Learning models are trained in practice

This project serves as an excellent introduction to modern Deep Learning workflows and prepares you for more advanced architectures such as Convolutional Neural Networks (CNNs), Transfer Learning, and Computer Vision systems.
