# Gradient Vanishing and Exploding Solutions

A NumPy-based neural network project for studying the **Vanishing Gradient** and **Exploding Gradient** problems in deep neural networks and evaluating different techniques to alleviate them.

The neural network is implemented from scratch using NumPy instead of deep learning frameworks such as TensorFlow or PyTorch.

---

## 📌 Project Overview

Deep neural networks may suffer from unstable gradient propagation during backpropagation.

Two common problems are:

- **Vanishing Gradient**: gradients become extremely small as they propagate toward earlier layers.
- **Exploding Gradient**: gradients become extremely large during backpropagation.

These problems can make deep networks difficult or impossible to train.

This project investigates several techniques that can improve gradient propagation, including:

- Different activation functions
- Different weight initialization methods
- Batch Normalization
- Gradient monitoring and visualization

The experiments are conducted on the **MNIST handwritten digit classification dataset**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a deep neural network from scratch using NumPy.
2. Reproduce the Vanishing/Exploding Gradient problem.
3. Monitor gradients across different layers during training.
4. Compare different solutions for improving gradient flow.
5. Evaluate the effect of each method on training performance.
6. Visualize gradient behavior, loss, and accuracy.

---

## 🧠 Neural Network Architecture

The default network is a deep Multi-Layer Perceptron (MLP).

```text
Input
  │
  │ 784 features
  ▼
Hidden Layer 1
  │ 64 neurons
  ▼
Hidden Layer 2
  │ 64 neurons
  ▼
   ...
  │
  ▼
Hidden Layer 19
  │ 64 neurons
  ▼
Output Layer
  │ 10 neurons
  ▼
Softmax
```

Default configuration:

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Input dimension | 784 |
| Hidden dimension | 64 |
| Output dimension | 10 |
| Number of weighted layers | 20 |
| Default hidden activation | Sigmoid |
| Output activation | Softmax |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 50 |
| Random seed | 42 |

---

## 📁 Project Structure

```text
Gradient-Vanishing-and-Solution/
│
├── data/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   ├── y_val.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── src/
│   ├── __init__.py
│   ├── activation.py
│   ├── initializers.py
│   ├── NeuralNetwork.py
│   └── visualize.py
│
├── config.py
├── Data_preparing.py
├── Main.ipynb
├── Task.md
├── requirements.txt
├── .gitignore
└── README.md
```

### File responsibilities

**`src/activation.py`**

Contains activation functions and their backward derivatives.

Currently supported:

```text
ReLU
Leaky ReLU
ELU
Sigmoid
Tanh
```

---

**`src/initializers.py`**

Contains different weight initialization strategies.

Currently supported:

```text
Random Normal
Xavier Normal
He Normal
He Normal for Leaky ReLU
```

---

**`src/NeuralNetwork.py`**

Contains the main `NeuralNetwork` class.

Responsibilities include:

```text
Parameter initialization
        │
        ▼
Forward propagation
        │
        ▼
Batch Normalization
        │
        ▼
Activation functions
        │
        ▼
Backward propagation
        │
        ▼
Gradient calculation
        │
        ▼
Parameter update
        │
        ▼
Prediction
```

---

**`src/visualize.py`**

Provides utilities for analyzing and visualizing experiment results.

Supported visualizations include:

- Gradient magnitude across layers
- Gradient evolution across epochs
- Gradient ratio between early and late layers
- Training/validation loss
- Training/validation accuracy
- Experiment comparison

The module also supports saving and loading experiment histories using `.npz` files.

---

**`config.py`**

Stores the common experiment configuration, including:

- Network architecture
- Activation configuration
- Learning rate
- Batch size
- Number of epochs
- Random seed
- Logging configuration

Using a centralized configuration helps ensure that different experiments are compared under similar conditions.

---

**`Data_preparing.py`**

Downloads and prepares the MNIST dataset.

The script:

```text
Download MNIST
      │
      ▼
Convert labels to integers
      │
      ▼
Normalize pixel values to [0, 1]
      │
      ▼
Split dataset
      │
      ├── 70% Training
      ├── 15% Validation
      └── 15% Testing
      │
      ▼
Save datasets as .npy files
```

---

**`Main.ipynb`**

The main experiment notebook.

It is used to:

- Load the prepared MNIST dataset
- Create neural network models
- Train different experiment pipelines
- Compare training results
- Analyze different solutions for gradient problems

---

## 🧪 Experiment Pipelines

The project can compare different combinations of activation functions, initialization strategies, and normalization methods.

An example experiment sequence is:

```text
Baseline
Sigmoid + Random Normal
          │
          ▼
Change Weight Initialization
Sigmoid + He Initialization
          │
          ▼
Change Activation
LeakyReLU + Random Normal
          │
          ▼
Combine Activation + Initialization
LeakyReLU + He Initialization
          │
          ▼
Add Normalization
LeakyReLU + He Initialization + BatchNorm
```

This allows the effect of each solution to be analyzed independently.

---

## 📊 Gradient Monitoring

Accuracy alone is not sufficient for studying the Vanishing Gradient problem.

Therefore, the project is designed to monitor gradient statistics for individual layers.

Important metrics include:

```text
dW_norm
dW_rms
dZ_norm
dZ_rms
```

For example:

```text
Gradient magnitude
       ▲
       │
Layer1 │ •
       │  •
       │    •
       │       •
       │             •
       └──────────────────────► Layer
          1   2   3   ...   20
```

A strong decrease in gradient magnitude toward the earlier layers can indicate a Vanishing Gradient problem.

---

## 📈 Visualization

The visualization module can compare multiple experiment histories.

Example:

```python
from src.visualize import plot_grad_by_layer

histories = {
    "Baseline": baseline_history,
    "He + LeakyReLU": he_history,
    "BatchNorm": batchnorm_history,
}

plot_grad_by_layer(histories)
```

Other available functions include:

```python
plot_grad_by_layer(...)
plot_grad_over_epochs(...)
plot_loss_over_epochs(...)
plot_acc_over_epochs(...)
plot_experiment_summary(...)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Gradient-Vanishing-and-Solution
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
NumPy
Matplotlib
scikit-learn
Jupyter
IPython Kernel
```

---

## 📥 Preparing the Dataset

Run:

```bash
python Data_preparing.py
```

The script will automatically download MNIST and create:

```text
data/
├── X_train.npy
├── y_train.npy
├── X_val.npy
├── y_val.npy
├── X_test.npy
└── y_test.npy
```

Expected dataset sizes:

```text
Training:   49,000 samples
Validation: 10,500 samples
Testing:    10,500 samples
```

The `data/` directory is excluded from Git because the dataset can be regenerated using `Data_preparing.py`.

---

## ▶️ Running Experiments

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Main.ipynb
```

Run the cells sequentially to:

```text
Load MNIST
    ↓
Create Model
    ↓
Train Model
    ↓
Evaluate Model
    ↓
Compare Experiments
```

---

## 🔬 Weight Initialization

### Random Normal

```text
W ~ N(0, 0.01²)
```

A very small initialization may contribute to vanishing activations and gradients in deep networks.

### Xavier Initialization

Designed to maintain a more stable variance through the network and is commonly suitable for activation functions such as `tanh`.

### He Initialization

Uses:

```text
std = sqrt(2 / fan_in)
```

and is commonly paired with ReLU-family activation functions.

### He Initialization for Leaky ReLU

Uses the negative slope of LeakyReLU when determining the initialization scale.

---

## ⚡ Activation Functions

The project separates activation functions into two groups.

### Saturating Activations

```text
Sigmoid
Tanh
```

These functions can saturate for large positive or negative inputs, causing their derivatives to become very small.

They are therefore useful for demonstrating the Vanishing Gradient problem.

### Non-Saturating Activations

```text
ReLU
Leaky ReLU
ELU
```

These activation functions generally allow gradients to propagate more effectively through deep networks.

---

## 🧮 Batch Normalization

Batch Normalization is implemented for hidden layers.

During training:

```text
Z
│
├── Calculate batch mean
│
├── Calculate batch variance
│
▼
Normalize
│
▼
Scale by gamma
│
▼
Shift by beta
│
▼
Activation
```

Running mean and variance are maintained for inference.

During prediction, the network uses these running statistics instead of statistics calculated from the current input batch.

---

## 📝 Current Development Status

The core components have been implemented:

```text
[✓] Activation functions
[✓] Activation derivatives
[✓] Weight initialization
[✓] Forward propagation
[✓] Backpropagation
[✓] Mini-batch training
[✓] Batch Normalization
[✓] MNIST preparation
[✓] Visualization utilities

[ ] Complete experiment history collection
[ ] Record gradient statistics during training
[ ] Integrate NeuralNetwork.fit() with visualize.py
[ ] Add gradient checking
[ ] Add automated tests
[ ] Add Gradient Clipping experiment
[ ] Complete experiment report
```

---

## 🚧 Planned Improvements

Future improvements include:

- Gradient history tracking
- Numerical gradient checking
- Gradient clipping
- More reproducible experiment pipelines
- Unit tests for activation functions
- Unit tests for weight initialization
- Unit tests for forward/backward propagation
- Automatic experiment result saving
- More detailed experiment comparisons

---

## 🛠️ Technologies

```text
Python
NumPy
scikit-learn
Matplotlib
Jupyter Notebook
```

The neural network itself is implemented manually using **NumPy**.

---

## 📚 Topic

**Research on solutions for Vanishing and Exploding Gradients in Deep Neural Networks implemented from scratch using NumPy.**

---

## 📄 License

See the `LICENSE` file for more information.