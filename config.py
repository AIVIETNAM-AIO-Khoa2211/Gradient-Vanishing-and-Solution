"""
Global configuration for the NumPy Neural Network project.

Topic:
    Vanishing / Exploding Gradient in Deep Neural Networks.

This file contains:
    - Project paths
    - Network architecture
    - Activation configurations
    - Training hyperparameters
    - Batch Normalization parameters
    - Gradient tracking configuration
    - Experiment configurations
"""

from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = ROOT_DIR / "logs"

HISTORY_DIR = RESULTS_DIR / "histories"
FIGURE_DIR = RESULTS_DIR / "figures"


# ============================================================
# RANDOM SEED
# ============================================================

# Use the same seed for all experiments so that comparisons
# are as fair and reproducible as possible.
SEED = 42


# ============================================================
# NETWORK ARCHITECTURE
# ============================================================

# Number of weighted layers.
#
# Input layer is NOT counted.
#
# Example:
#
#   input
#     ↓
#   layer 1
#     ↓
#   layer 2
#     ↓
#    ...
#     ↓
#   layer 20
#
NUM_LAYERS = 20

# MNIST:
# 28 x 28 = 784 input features
INPUT_DIM = 784

# Hidden neurons per hidden layer
HIDDEN_DIM = 64

# MNIST has 10 digit classes
OUTPUT_DIM = 10


# ------------------------------------------------------------
# Layer dimensions
# ------------------------------------------------------------
#
# Length:
#
#     NUM_LAYERS + 1
#
# Example:
#
#     [
#         784,
#         64,
#         64,
#         ...,
#         10
#     ]
#
LAYER_SIZES = [INPUT_DIM] + [HIDDEN_DIM] * (NUM_LAYERS - 1) + [OUTPUT_DIM]


# ============================================================
# ACTIVATION CONFIGURATIONS
# ============================================================

# ------------------------------------------------------------
# Baseline
#
# Deep Sigmoid network is intentionally used to demonstrate
# the Vanishing Gradient problem.
# ------------------------------------------------------------

SIGMOID_ACTIVATIONS = [None] + ["sigmoid"] * (NUM_LAYERS - 1) + ["softmax"]


# ------------------------------------------------------------
# ReLU
# ------------------------------------------------------------

RELU_ACTIVATIONS = [None] + ["relu"] * (NUM_LAYERS - 1) + ["softmax"]


# ------------------------------------------------------------
# Leaky ReLU
# ------------------------------------------------------------

LEAKY_RELU_ACTIVATIONS = (
    [None] + ["leaky_relu"] * (NUM_LAYERS - 1) + ["softmax"]
)


# ------------------------------------------------------------
# ELU
# ------------------------------------------------------------

ELU_ACTIVATIONS = [None] + ["elu"] * (NUM_LAYERS - 1) + ["softmax"]


# ------------------------------------------------------------
# Tanh
# ------------------------------------------------------------

TANH_ACTIVATIONS = [None] + ["tanh"] * (NUM_LAYERS - 1) + ["softmax"]


# ============================================================
# BACKWARD-COMPATIBILITY ALIASES
# ============================================================

# Old code in Main.ipynb may still use these names.
#
# They are kept temporarily so the current notebook does not break.

ACTIVATIONS = SIGMOID_ACTIVATIONS

PIPELINE1 = LEAKY_RELU_ACTIVATIONS


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

LEARNING_RATE = 0.01

BATCH_SIZE = 64

NUM_EPOCHS = 50

PRINT_EVERY = 10


# ============================================================
# BATCH NORMALIZATION
# ============================================================

# Running statistics update:
#
# running_mean =
#     momentum * running_mean
#     + (1 - momentum) * batch_mean
#
BN_MOMENTUM = 0.9

# Numerical stability constant.
BN_EPSILON = 1e-5


# ============================================================
# GRADIENT TRACKING
# ============================================================

# Save gradient statistics for each layer.
SAVE_GRADIENT_NORMS = True

# Metrics currently produced by NeuralNetwork.fit().
GRADIENT_METRICS = (
    "dW_norm",
    "dW_rms",
    "dZ_norm",
    "dZ_rms",
)


# ============================================================
# ACTIVATION STATISTICS
# ============================================================

# Reserved for future activation-distribution experiments.
#
# The current NeuralNetwork implementation does not yet save
# activation mean / variance into history.
SAVE_ACTIVATION_STATS = False


# ============================================================
# GRADIENT CLIPPING
# ============================================================

# Gradient clipping has not yet been implemented in
# NeuralNetwork.py.
#
# Keep it disabled until the clipping experiment is added.

USE_GRADIENT_CLIPPING = False

GRADIENT_CLIP_NORM = None


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

# These configurations correspond to the experiments currently
# being studied in Main.ipynb.
#
# Keeping experiment definitions here avoids repeatedly
# hard-coding activation / initializer / BatchNorm combinations
# inside the notebook.

EXPERIMENTS = {
    # --------------------------------------------------------
    # Experiment 1
    #
    # Baseline:
    # Sigmoid + small Random Normal initialization
    # --------------------------------------------------------
    "baseline": {
        "name": "Baseline: Sigmoid + Random Normal",
        "activations": SIGMOID_ACTIVATIONS,
        "initializer": "random_normal",
        "use_batchnorm": False,
    },
    # --------------------------------------------------------
    # Experiment 2
    #
    # Only change initialization.
    # --------------------------------------------------------
    "he_sigmoid": {
        "name": "Sigmoid + He Initialization",
        "activations": SIGMOID_ACTIVATIONS,
        "initializer": "he_normal",
        "use_batchnorm": False,
    },
    # --------------------------------------------------------
    # Experiment 3
    #
    # Only change activation.
    # --------------------------------------------------------
    "leaky_random": {
        "name": "LeakyReLU + Random Normal",
        "activations": LEAKY_RELU_ACTIVATIONS,
        "initializer": "random_normal",
        "use_batchnorm": False,
    },
    # --------------------------------------------------------
    # Experiment 4
    #
    # Combine non-saturating activation with He initialization.
    # --------------------------------------------------------
    "leaky_he": {
        "name": "LeakyReLU + He Initialization",
        "activations": LEAKY_RELU_ACTIVATIONS,
        "initializer": "he_normal_leaky",
        "use_batchnorm": False,
    },
    # --------------------------------------------------------
    # Experiment 5
    #
    # Add Batch Normalization.
    # --------------------------------------------------------
    "batchnorm": {
        "name": "Basline + BatchNorm",
        "activations": SIGMOID_ACTIVATIONS,
        "initializer": "random_normal",
        "use_batchnorm": True,
    },
}


# ============================================================
# DEFAULT EXPERIMENT ORDER
# ============================================================

EXPERIMENT_ORDER = (
    "baseline",
    "he_sigmoid",
    "leaky_random",
    "leaky_he",
    "batchnorm",
)
