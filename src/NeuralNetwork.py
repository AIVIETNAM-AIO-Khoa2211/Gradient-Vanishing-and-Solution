import numpy as np
from src.activation import NonSaturatingActivations, SaturatingActivations
from src.initializers import WeightInitializer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=0, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=0, keepdims=True)


class NeuralNetwork:

    def __init__(
        self,
        layer_dims,
        activations=None,
        seed=42,
        Weights_initializer=None,
        use_batchnorm=True,
    ):
        self.layer_dims = layer_dims
        self.n_layer = len(layer_dims) - 1
        self.Weights_initializer = Weights_initializer
        self.seed = seed

        self.use_batchnorm = use_batchnorm
        self.bn_momentum = 0.9  # Dùng để cập nhật running_mean/var
        self.bn_eps = 1e-5  # Chống chia cho 0

        self._ACTIVATION_MAP = {
            "relu": NonSaturatingActivations.relu,
            "leaky_relu": NonSaturatingActivations.leaky_relu,
            "elu": NonSaturatingActivations.elu,
            "sigmoid": SaturatingActivations.sigmoid,
            "tanh": SaturatingActivations.tanh,
            "softmax": lambda z: (
                softmax(z),
                lambda dout: dout,
            ),  # softmax + cross-entropy
            "none": lambda z: (z, lambda dout: dout),
        }

        if activations is None:
            self.activations = (
                ["none"] + ["relu"] * (self.n_layer - 1) + ["softmax"]
            )
        elif len(activations) == self.n_layer:
            self.activations = ["none"] + activations
        elif len(activations) == self.n_layer + 1:
            self.activations = activations
        else:
            raise ValueError(
                f"activations must have length {self.n_layer} "
                f"or {self.n_layer + 1}"
            )

        self._initialize_param()

    def _initialize_param(self):
        initializer = WeightInitializer(
            self.layer_dims, method=self.Weights_initializer, seed=self.seed
        )
        self.W, self.b = initializer.initialize()

        if self.use_batchnorm:
            self.gamma = {}
            self.beta = {}
            self.running_mean = {}
            self.running_var = {}

            # BN chỉ áp dụng cho hidden layers (i từ 1 đến n_layer - 1)
            for i in range(1, self.n_layer):
                n_out = self.layer_dims[i]
                self.gamma[i] = np.ones((n_out, 1))  # gamma khởi tạo là 1
                self.beta[i] = np.zeros((n_out, 1))  # beta khởi tạo là 0
                self.running_mean[i] = np.zeros((n_out, 1))
                self.running_var[i] = np.zeros((n_out, 1))

    def _activate(self, Z, name):
        """Apply the named HIDDEN-layer activation to Z. Not for softmax (output layer)."""
        if name not in self._ACTIVATION_MAP:
            raise ValueError(f"Unknown activation function: {name}")
        return self._ACTIVATION_MAP[name](Z)

    def _forward(self, X, mode="train"):
        Z = {}
        Z_norm = {}  # Lưu lại Z_norm để dùng cho backward
        A = {0: X}
        backward = {}
        bn_cache = {}  # Cache riêng cho BatchNorm

        for i in range(1, self.n_layer + 1):
            Z[i] = self.W[i] @ A[i - 1] + self.b[i]

            # Áp dụng BatchNorm cho các lớp ẩn
            if self.use_batchnorm and i < self.n_layer:
                if mode == "train":
                    # 1. Tính Mean và Variance của mini-batch
                    mu = np.mean(Z[i], axis=1, keepdims=True)
                    var = np.var(Z[i], axis=1, keepdims=True)

                    # 2. Chuẩn hóa
                    Z_norm[i] = (Z[i] - mu) / np.sqrt(var + self.bn_eps)

                    # 3. Scale & Shift
                    Z_bn = self.gamma[i] * Z_norm[i] + self.beta[i]

                    # 4. Lưu cache và cập nhật Running stats
                    bn_cache[i] = (Z_norm[i], var, mu)
                    self.running_mean[i] = (
                        self.bn_momentum * self.running_mean[i]
                        + (1 - self.bn_momentum) * mu
                    )
                    self.running_var[i] = (
                        self.bn_momentum * self.running_var[i]
                        + (1 - self.bn_momentum) * var
                    )
                else:  # mode == 'test'
                    # Dùng running stats khi dự đoán
                    Z_norm_test = (Z[i] - self.running_mean[i]) / np.sqrt(
                        self.running_var[i] + self.bn_eps
                    )
                    Z_bn = self.gamma[i] * Z_norm_test + self.beta[i]

                # Đưa Z_bn qua hàm kích hoạt thay vì Z[i]
                A[i], backward[i] = self._activate(Z_bn, self.activations[i])
            else:
                A[i], backward[i] = self._activate(Z[i], self.activations[i])

        return A[self.n_layer], {
            "Z": Z,
            "A": A,
            "backward": backward,
            "bn_cache": bn_cache,
        }

    def _backward(self, y, cache):
        y_ohe = self.encoder.transform(y.reshape(-1, 1)).T
        n_samples = y.shape[0]
        dZ = {}
        dW = {}
        db = {}

        # Khai báo biến lưu gradient của gamma và beta
        dgamma = {}
        dbeta = {}

        dZ[self.n_layer] = cache["A"][self.n_layer] - y_ohe

        for i in range(self.n_layer, 0, -1):
            if self.use_batchnorm and i < self.n_layer:
                # Lấy cache của BatchNorm
                Z_norm, var, mu = cache["bn_cache"][i]

                # Tính gradient cho gamma và beta
                dgamma[i] = np.sum(dZ[i] * Z_norm, axis=1, keepdims=True)
                dbeta[i] = np.sum(dZ[i], axis=1, keepdims=True)

                # Tính dZ mới truyền ngược qua cục BatchNorm
                dZ_norm = dZ[i] * self.gamma[i]
                std_inv = 1.0 / np.sqrt(var + self.bn_eps)

                # Công thức đạo hàm phức tạp của BatchNorm (truyền ngược từ Z_norm về Z)
                dZ[i] = (
                    (1.0 / n_samples)
                    * std_inv
                    * (
                        n_samples * dZ_norm
                        - np.sum(dZ_norm, axis=1, keepdims=True)
                        - Z_norm
                        * np.sum(dZ_norm * Z_norm, axis=1, keepdims=True)
                    )
                )

            dW[i] = (1 / n_samples) * dZ[i] @ cache["A"][i - 1].T
            db[i] = (1 / n_samples) * np.sum(dZ[i], axis=1, keepdims=True)

            if i > 1:
                dA = self.W[i].T @ dZ[i]
                dZ[i - 1] = cache["backward"][i - 1](dA)

        return {"dW": dW, "db": db, "dgamma": dgamma, "dbeta": dbeta}

    def _update(self, grads, learning_rate):
        for i in range(1, self.n_layer + 1):
            self.W[i] -= learning_rate * grads["dW"][i]
            self.b[i] -= learning_rate * grads["db"][i]

            if self.use_batchnorm and i < self.n_layer:
                self.gamma[i] -= learning_rate * grads["dgamma"][i]
                self.beta[i] -= learning_rate * grads["dbeta"][i]

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        learning_rate=0.1,
        epochs=1000,
        batch_size=64,
        print_every=10,
    ):
        n_samples = X.shape[1]  # feature-major: (features, samples)

        # Initialize OneHotEncoder for y labels
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(y.reshape(-1, 1))

        for epoch in range(epochs):
            # Shuffle mỗi epoch
            perm = np.random.permutation(n_samples)
            X_shuffled = X[:, perm]
            y_shuffled = y[perm]

            # Loop qua từng mini-batch
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]

                # 1. THÊM mode='train' VÀO ĐÂY
                # Để BatchNorm tính toán mean/var trên batch này và cập nhật running stats
                A_L, cache = self._forward(X_batch, mode="train")

                grads = self._backward(y_batch, cache)
                self._update(grads, learning_rate)

            if epoch % print_every == 0:
                # 2. THÊM mode='test' VÀO ĐÂY
                # Đánh giá trên toàn bộ tập X phải dùng running_mean và running_var đã lưu
                A_L_full, _ = self._forward(X, mode="test")
                predictions = np.argmax(A_L_full, axis=0)

                print("Epoch:", epoch)
                print("Accuracy:", accuracy_score(predictions, y))

                if X_val is not None and y_val is not None:
                    # Hàm predict (nếu bạn đã sửa như ở phần trước) cũng sẽ tự gọi mode='test'
                    print(
                        f"val_acc: {accuracy_score(self.predict(X_val), y_val):.4f}"
                    )

    def predict(self, X):
        A_L, _ = self._forward(X, mode="test")  # CHÚ Ý TRUYỀN mode='test'
        return np.argmax(A_L, axis=0)
