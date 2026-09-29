import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from src.activation import NonSaturatingActivations, SaturatingActivations
from src.initializers import WeightInitializer

EPS = 1e-12


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.

    Parameters
    ----------
    x : np.ndarray
        Shape: (n_classes, n_samples)

    Returns
    -------
    np.ndarray
        Softmax probabilities with the same shape as x.
    """
    x_max = np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


class NeuralNetwork:
    def __init__(
        self,
        layer_dims,
        activations=None,
        seed=42,
        Weights_initializer="random_normal",
        use_batchnorm=True,
    ):
        """
        Fully-connected Neural Network implemented from scratch using NumPy.

        Parameters
        ----------
        layer_dims : list[int]
            Example:
            [784, 64, 64, ..., 10]

        activations : list[str] | None
            Activation functions for each layer.

            Can have:
                len(layer_dims) - 1
            or:
                len(layer_dims)

        seed : int
            Random seed used for initialization and data shuffling.

        Weights_initializer : str
            Weight initialization method.

            Supported:
                random_normal
                xavier_normal
                he_normal
                he_normal_leaky

        use_batchnorm : bool
            Whether Batch Normalization is applied to hidden layers.
        """
        self.layer_dims = layer_dims
        self.n_layer = len(layer_dims) - 1

        self.Weights_initializer = Weights_initializer
        self.seed = seed

        # RNG riêng của model để các experiment có thể tái lập.
        self.rng = np.random.default_rng(seed)

        self.use_batchnorm = use_batchnorm

        # Batch Normalization hyperparameters
        self.bn_momentum = 0.9
        self.bn_eps = 1e-5

        # Activation function mapping
        self._ACTIVATION_MAP = {
            "relu": NonSaturatingActivations.relu,
            "leaky_relu": NonSaturatingActivations.leaky_relu,
            "elu": NonSaturatingActivations.elu,
            "sigmoid": SaturatingActivations.sigmoid,
            "tanh": SaturatingActivations.tanh,
            "softmax": lambda z: (
                softmax(z),
                lambda dout: dout,
            ),
            "none": lambda z: (
                z,
                lambda dout: dout,
            ),
        }

        # -----------------------------------------------------
        # Activation configuration
        # -----------------------------------------------------
        if activations is None:
            self.activations = (
                ["none"] + ["relu"] * (self.n_layer - 1) + ["softmax"]
            )

        elif len(activations) == self.n_layer:
            self.activations = ["none"] + list(activations)

        elif len(activations) == self.n_layer + 1:
            self.activations = list(activations)

        else:
            raise ValueError(
                f"activations must have length {self.n_layer} "
                f"or {self.n_layer + 1}"
            )

        self._initialize_param()

        # OneHotEncoder sẽ được khởi tạo trong fit()
        self.encoder = None

        # Lưu history sau khi train
        self.history_ = None

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_param(self):
        initializer = WeightInitializer(
            self.layer_dims,
            method=self.Weights_initializer,
            seed=self.seed,
        )

        self.W, self.b = initializer.initialize()

        if self.use_batchnorm:
            self.gamma = {}
            self.beta = {}

            self.running_mean = {}
            self.running_var = {}

            # BatchNorm chỉ áp dụng cho hidden layers
            for i in range(1, self.n_layer):
                n_out = self.layer_dims[i]

                self.gamma[i] = np.ones((n_out, 1))
                self.beta[i] = np.zeros((n_out, 1))

                self.running_mean[i] = np.zeros((n_out, 1))
                self.running_var[i] = np.zeros((n_out, 1))

    # ============================================================
    # ACTIVATION
    # ============================================================

    def _activate(self, Z, name):
        """
        Apply an activation function.

        Returns
        -------
        A : np.ndarray
            Activation output.

        backward : callable
            Function used during backpropagation.
        """
        if name not in self._ACTIVATION_MAP:
            raise ValueError(f"Unknown activation function: {name}")

        return self._ACTIVATION_MAP[name](Z)

    # ============================================================
    # FORWARD PROPAGATION
    # ============================================================

    def _forward(self, X, mode="train"):
        """
        Forward propagation.

        Parameters
        ----------
        X : np.ndarray
            Shape: (n_features, n_samples)

        mode : str
            "train" or "test"

        Returns
        -------
        output : np.ndarray
            Output probabilities.

        cache : dict
            Values required for backpropagation.
        """
        if mode not in {"train", "test"}:
            raise ValueError("mode must be either 'train' or 'test'")

        Z = {}
        A = {0: X}

        backward = {}
        bn_cache = {}

        for i in range(1, self.n_layer + 1):

            # Linear transformation
            Z[i] = self.W[i] @ A[i - 1] + self.b[i]

            # --------------------------------------------------
            # Batch Normalization
            # --------------------------------------------------
            if self.use_batchnorm and i < self.n_layer:

                if mode == "train":

                    # Batch statistics
                    mu = np.mean(
                        Z[i],
                        axis=1,
                        keepdims=True,
                    )

                    var = np.var(
                        Z[i],
                        axis=1,
                        keepdims=True,
                    )

                    # Normalize
                    Z_norm = (Z[i] - mu) / np.sqrt(var + self.bn_eps)

                    # Scale and shift
                    Z_bn = self.gamma[i] * Z_norm + self.beta[i]

                    # Cache for backward
                    bn_cache[i] = {
                        "Z_norm": Z_norm,
                        "var": var,
                        "mu": mu,
                    }

                    # Update running statistics
                    self.running_mean[i] = (
                        self.bn_momentum * self.running_mean[i]
                        + (1.0 - self.bn_momentum) * mu
                    )

                    self.running_var[i] = (
                        self.bn_momentum * self.running_var[i]
                        + (1.0 - self.bn_momentum) * var
                    )

                else:

                    # During inference use running statistics
                    Z_norm = (Z[i] - self.running_mean[i]) / np.sqrt(
                        self.running_var[i] + self.bn_eps
                    )

                    Z_bn = self.gamma[i] * Z_norm + self.beta[i]

                A[i], backward[i] = self._activate(
                    Z_bn,
                    self.activations[i],
                )

            else:
                A[i], backward[i] = self._activate(
                    Z[i],
                    self.activations[i],
                )

        cache = {
            "Z": Z,
            "A": A,
            "backward": backward,
            "bn_cache": bn_cache,
        }

        return A[self.n_layer], cache

    # ============================================================
    # BACKPROPAGATION
    # ============================================================

    def _backward(self, y, cache):
        """
        Backpropagation.

        Returns gradients for:
            W
            b
            gamma
            beta
            Z
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder has not been initialized. "
                "Call fit() before backward propagation."
            )

        y_ohe = self.encoder.transform(y.reshape(-1, 1)).T

        n_samples = y.shape[0]

        dZ = {}
        dW = {}
        db = {}

        dgamma = {}
        dbeta = {}

        # ------------------------------------------------------
        # Softmax + Cross Entropy
        #
        # derivative:
        #
        # dZ_L = y_hat - y
        # ------------------------------------------------------
        dZ[self.n_layer] = cache["A"][self.n_layer] - y_ohe

        for i in range(
            self.n_layer,
            0,
            -1,
        ):

            # --------------------------------------------------
            # Backward through BatchNorm
            # --------------------------------------------------
            if self.use_batchnorm and i < self.n_layer:

                bn_cache = cache["bn_cache"][i]

                Z_norm = bn_cache["Z_norm"]
                var = bn_cache["var"]

                # Gradient gamma / beta
                #
                # Loss sử dụng mean trên batch nên cần / n_samples.
                dgamma[i] = (
                    np.sum(
                        dZ[i] * Z_norm,
                        axis=1,
                        keepdims=True,
                    )
                    / n_samples
                )

                dbeta[i] = (
                    np.sum(
                        dZ[i],
                        axis=1,
                        keepdims=True,
                    )
                    / n_samples
                )

                # Backpropagate through gamma
                dZ_norm = dZ[i] * self.gamma[i]

                std_inv = 1.0 / np.sqrt(var + self.bn_eps)

                # BatchNorm derivative
                dZ[i] = (
                    (1.0 / n_samples)
                    * std_inv
                    * (
                        n_samples * dZ_norm
                        - np.sum(
                            dZ_norm,
                            axis=1,
                            keepdims=True,
                        )
                        - Z_norm
                        * np.sum(
                            dZ_norm * Z_norm,
                            axis=1,
                            keepdims=True,
                        )
                    )
                )

            # --------------------------------------------------
            # Weight and bias gradients
            # --------------------------------------------------
            dW[i] = (dZ[i] @ cache["A"][i - 1].T) / n_samples

            db[i] = (
                np.sum(
                    dZ[i],
                    axis=1,
                    keepdims=True,
                )
                / n_samples
            )

            # --------------------------------------------------
            # Gradient propagated to previous layer
            # --------------------------------------------------
            if i > 1:

                dA_prev = self.W[i].T @ dZ[i]

                dZ[i - 1] = cache["backward"][i - 1](dA_prev)

        return {
            "dW": dW,
            "db": db,
            "dZ": dZ,
            "dgamma": dgamma,
            "dbeta": dbeta,
        }

    # ============================================================
    # UPDATE PARAMETERS
    # ============================================================

    def _update(self, grads, learning_rate):
        for i in range(
            1,
            self.n_layer + 1,
        ):

            self.W[i] -= learning_rate * grads["dW"][i]

            self.b[i] -= learning_rate * grads["db"][i]

            if self.use_batchnorm and i < self.n_layer:
                self.gamma[i] -= learning_rate * grads["dgamma"][i]

                self.beta[i] -= learning_rate * grads["dbeta"][i]

    # ============================================================
    # LOSS
    # ============================================================

    def _cross_entropy_loss(self, probabilities, y):
        """
        Mean categorical cross-entropy.

        Parameters
        ----------
        probabilities : np.ndarray
            Shape: (n_classes, n_samples)

        y : np.ndarray
            Shape: (n_samples,)
        """
        y_ohe = self.encoder.transform(y.reshape(-1, 1)).T

        probabilities = np.clip(
            probabilities,
            EPS,
            1.0,
        )

        loss = -np.sum(y_ohe * np.log(probabilities)) / y.shape[0]

        return float(loss)

    # ============================================================
    # GRADIENT STATISTICS
    # ============================================================

    def _gradient_statistics(self, grads):
        """
        Calculate gradient statistics layer by layer.

        Returns
        -------
        dict with:
            dW_norm
            dW_rms
            dZ_norm
            dZ_rms
        """
        dW_norm = np.zeros(
            self.n_layer,
            dtype=float,
        )

        dW_rms = np.zeros(
            self.n_layer,
            dtype=float,
        )

        dZ_norm = np.zeros(
            self.n_layer,
            dtype=float,
        )

        dZ_rms = np.zeros(
            self.n_layer,
            dtype=float,
        )

        for i in range(
            1,
            self.n_layer + 1,
        ):

            dW = grads["dW"][i]
            dZ = grads["dZ"][i]

            # L2 norm
            dW_norm[i - 1] = np.linalg.norm(dW)
            dZ_norm[i - 1] = np.linalg.norm(dZ)

            # RMS gradient
            dW_rms[i - 1] = np.sqrt(np.mean(np.square(dW)))

            dZ_rms[i - 1] = np.sqrt(np.mean(np.square(dZ)))

        return {
            "dW_norm": dW_norm,
            "dW_rms": dW_rms,
            "dZ_norm": dZ_norm,
            "dZ_rms": dZ_rms,
        }

    # ============================================================
    # EVALUATION
    # ============================================================

    def _evaluate(self, X, y):
        """
        Evaluate loss and accuracy using inference mode.
        """
        probabilities, _ = self._forward(
            X,
            mode="test",
        )

        predictions = np.argmax(
            probabilities,
            axis=0,
        )

        loss = self._cross_entropy_loss(
            probabilities,
            y,
        )

        accuracy = accuracy_score(
            y,
            predictions,
        )

        return loss, accuracy

    # ============================================================
    # TRAINING
    # ============================================================

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        learning_rate=0.1,
        epochs=100,
        batch_size=64,
        print_every=10,
    ):
        """
        Train the neural network.

        Returns
        -------
        history : dict

        Example keys:

            history["epoch"]
            history["train_loss"]
            history["val_loss"]
            history["train_acc"]
            history["val_acc"]

            history["dW_norm"]
            history["dW_rms"]
            history["dZ_norm"]
            history["dZ_rms"]

        Gradient arrays have shape:

            (epochs, n_layers)
        """

        if epochs <= 0:
            raise ValueError("epochs must be greater than 0")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if print_every <= 0:
            raise ValueError("print_every must be greater than 0")

        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val and y_val must either both be provided "
                "or both be None."
            )

        n_samples = X.shape[1]

        if y.shape[0] != n_samples:
            raise ValueError("X and y contain a different number of samples.")

        # -------------------------------------------------------
        # One-hot encoder
        # -------------------------------------------------------
        self.encoder = OneHotEncoder(sparse_output=False)

        self.encoder.fit(y.reshape(-1, 1))

        # -------------------------------------------------------
        # History
        # -------------------------------------------------------
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "dW_norm": [],
            "dW_rms": [],
            "dZ_norm": [],
            "dZ_rms": [],
        }

        if X_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        # =======================================================
        # TRAINING LOOP
        # =======================================================

        for epoch in range(epochs):

            # ---------------------------------------------------
            # Reproducible shuffle
            # ---------------------------------------------------
            permutation = self.rng.permutation(n_samples)

            X_shuffled = X[:, permutation]
            y_shuffled = y[permutation]

            # Gradient statistics accumulated across mini-batches
            epoch_dW_norm = np.zeros(self.n_layer)

            epoch_dW_rms = np.zeros(self.n_layer)

            epoch_dZ_norm = np.zeros(self.n_layer)

            epoch_dZ_rms = np.zeros(self.n_layer)

            n_batches = 0

            # ---------------------------------------------------
            # MINI-BATCH TRAINING
            # ---------------------------------------------------
            for start in range(
                0,
                n_samples,
                batch_size,
            ):

                end = min(
                    start + batch_size,
                    n_samples,
                )

                X_batch = X_shuffled[
                    :,
                    start:end,
                ]

                y_batch = y_shuffled[start:end]

                # Forward
                _, cache = self._forward(
                    X_batch,
                    mode="train",
                )

                # Backward
                grads = self._backward(
                    y_batch,
                    cache,
                )

                # -----------------------------------------------
                # Gradient statistics BEFORE parameter update
                # -----------------------------------------------
                grad_stats = self._gradient_statistics(grads)

                epoch_dW_norm += grad_stats["dW_norm"]

                epoch_dW_rms += grad_stats["dW_rms"]

                epoch_dZ_norm += grad_stats["dZ_norm"]

                epoch_dZ_rms += grad_stats["dZ_rms"]

                n_batches += 1

                # Update
                self._update(
                    grads,
                    learning_rate,
                )

            # ---------------------------------------------------
            # Average gradient statistics for this epoch
            # ---------------------------------------------------
            epoch_dW_norm /= n_batches
            epoch_dW_rms /= n_batches

            epoch_dZ_norm /= n_batches
            epoch_dZ_rms /= n_batches

            # ---------------------------------------------------
            # Evaluate training set
            # ---------------------------------------------------
            train_loss, train_acc = self._evaluate(
                X,
                y,
            )

            # ---------------------------------------------------
            # Store history
            # ---------------------------------------------------
            history["epoch"].append(epoch)

            history["train_loss"].append(train_loss)

            history["train_acc"].append(train_acc)

            history["dW_norm"].append(epoch_dW_norm)

            history["dW_rms"].append(epoch_dW_rms)

            history["dZ_norm"].append(epoch_dZ_norm)

            history["dZ_rms"].append(epoch_dZ_rms)

            # ---------------------------------------------------
            # Validation
            # ---------------------------------------------------
            if X_val is not None:

                val_loss, val_acc = self._evaluate(
                    X_val,
                    y_val,
                )

                history["val_loss"].append(val_loss)

                history["val_acc"].append(val_acc)

            # ---------------------------------------------------
            # Console output
            # ---------------------------------------------------
            if epoch % print_every == 0:

                print(f"Epoch: {epoch}")

                print(f"Train Loss: " f"{train_loss:.6f}")

                print(f"Train Accuracy: " f"{train_acc:.4f}")

                if X_val is not None:

                    print(f"Val Loss: " f"{val_loss:.6f}")

                    print(f"Val Accuracy: " f"{val_acc:.4f}")

                print("-" * 40)

        # =======================================================
        # Convert lists to NumPy arrays
        # =======================================================

        for key in history:
            history[key] = np.asarray(history[key])

        # Legacy alias supported by visualize.py
        history["loss"] = history["train_loss"]

        self.history_ = history

        return history

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict_proba(self, X):
        """
        Return class probabilities.

        Shape:
            (n_classes, n_samples)
        """
        probabilities, _ = self._forward(
            X,
            mode="test",
        )

        return probabilities

    def predict(self, X):
        """
        Return predicted class indices.

        Shape:
            (n_samples,)
        """
        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=0,
        )
