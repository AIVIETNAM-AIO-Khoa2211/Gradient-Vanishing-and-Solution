"""
config.py
Tham số cố định cho project NumPy Neural Network (Vanishing/Exploding Gradient).
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

# ============================================================
# ARCHITECTURE (baseline: MLP 20 layer)
# ============================================================
NUM_LAYERS = 20  # số layer (không tính input layer)

# LAYER_SIZES có độ dài NUM_LAYERS + 1 (bao gồm input dim ở vị trí [0])
# index 0  = input dim
# index i  = số neuron của layer thứ i (i = 1..NUM_LAYERS)
INPUT_DIM = 784   # MNIST flatten 28x28
OUTPUT_DIM = 10   # 10 class
HIDDEN_DIM = 64   # số neuron mặc định mỗi hidden layer

LAYER_SIZES = [INPUT_DIM] + [HIDDEN_DIM] * (NUM_LAYERS - 1) + [OUTPUT_DIM]

# ACTIVATIONS cùng convention 1-indexed như LAYER_SIZES (index 0 = None cho input layer)
# index 1..NUM_LAYERS-1 = hidden activation (mặc định sigmoid để dễ gây vanishing)
# index NUM_LAYERS      = output activation (softmax)
ACTIVATIONS = [None] + ["sigmoid"] * (NUM_LAYERS - 1) + ["softmax"]

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_EPOCHS = 50
SEED = 42  # cố định để so sánh công bằng giữa các solution

# ============================================================
# WEIGHT INITIALIZATION
# ============================================================
# Chọn phương pháp khởi tạo:  "random" (để test), "xavier" (cho sigmoid/tanh), "he" (cho relu), "he_leaky"
INIT_METHOD = "xavier"
# thêm negative slope nếu dùng He Initialization nếu dùng LeakyReLU
NEGATIVE_SLOPE = 0.01

#Điền vào nếu có những tham số cố định của phương pháp khởi tạo trọng số (weight initialization) cần sử dụng.

# ============================================================
# BATCH NORMALIZATION
# ============================================================

#Điền vào nếu có những tham số cố định của phương pháp chuẩn hóa theo batch (batch normalization) cần sử dụng.

# ============================================================
# GRADIENT CLIPPING
# ============================================================


# ============================================================
# LOGGING / EXPERIMENT TRACKING
# ============================================================
LOG_DIR = ROOT_DIR / "logs"
SAVE_GRADIENT_NORMS = True     # lưu gradient norm theo layer mỗi epoch
SAVE_ACTIVATION_STATS = True   # lưu mean/variance activation theo layer
