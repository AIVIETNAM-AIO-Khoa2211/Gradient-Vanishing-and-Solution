import numpy as np
from activation import NonSaturatingActivations, SaturatingActivations
from initializers import WeightInitializer


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=0, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=0, keepdims=True)


def one_hot_encode(y: np.ndarray) -> np.ndarray:
    y_encode = np.zeros([y.size, y.max() + 1])
    y_encode[np.arange(y.size), y] = 1
    return y_encode.T


def get_pred(output_layer: np.ndarray):
    return np.argmax(output_layer, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


class NeuralNetwork:
    ACTIVATION_MAP = {
                "relu": NonSaturatingActivations.relu,
                "leaky_relu": NonSaturatingActivations.leaky_relu,
                "elu": NonSaturatingActivations.elu,
                "sigmoid": SaturatingActivations.sigmoid,
                "tanh": SaturatingActivations.tanh,
                "none": lambda z: (z, lambda dout: dout),
                }

    def __init__(self, layer_dims, activations=None, seed=42):
        self.layer_dims = layer_dims
        self.n_layer = len(layer_dims) - 1
        self.seed = seed

        if activations is None:
            self.activations = ["none"] + ["relu"] * (self.n_layer - 1) + ["softmax"]
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
        rng = np.random.default_rng(self.seed)
        self.W = {}
        self.b = {}

        for i in range(1, self.n_layer + 1):
            self.W[i] = rng.random(
                (self.layer_dims[i], self.layer_dims[i - 1])
            ) - 0.5
            self.b[i] = np.zeros((self.layer_dims[i], 1))

    def _activate(self, Z, name):
        """Apply the named HIDDEN-layer activation to Z. Not for softmax (output layer)."""
        if name not in self._ACTIVATION_MAP:
            raise ValueError(f"Unknown activation function: {name}")
        return self._ACTIVATION_MAP[name](Z)

    def _forward(self, X):
        Z = {}
        A = {0: X}
        backward = {}

        for i in range(1, self.n_layer + 1):
            Z[i] = self.W[i] @ A[i - 1] + self.b[i]
            A[i], backward[i] = self._activate(Z[i], self.activations[i])

        return A[self.n_layer], {"Z": Z, "A": A, "backward": backward}

    def _backward(self, y, cache):
        y_ohe = one_hot_encode(y)
        n_samples = y.shape[0]
        dZ = {}
        dW = {}
        db = {}

        dZ[self.n_layer] = cache["A"][self.n_layer] - y_ohe

        for i in range(self.n_layer, 0, -1):
            dW[i] = (1 / n_samples) * dZ[i] @ cache["A"][i - 1].T
            db[i] = (1 / n_samples) * np.sum(dZ[i], axis=1, keepdims=True)

            if i > 1:
                dA = self.W[i].T @ dZ[i]
                dZ[i - 1] = cache["backward"][i - 1](dA)

        return {"dW": dW, "db": db}

    def _update(self, grads, learning_rate):
        for i in range(1, self.n_layer + 1):
            self.W[i] -= learning_rate * grads["dW"][i]
            self.b[i] -= learning_rate * grads["db"][i]

    def fit(self, X, y, X_val=None, y_val=None,
            learning_rate=0.1, epochs=1000, print_every=10):
        for i in range(epochs):
            A_L, cache = self._forward(X)
            grads = self._backward(y, cache)
            self._update(grads, learning_rate)

            if i % print_every == 0:
                predictions = get_pred(A_L)
                print("Iteration:", i)
                print("Accuracy:", get_accuracy(predictions, y))

                if X_val is not None and y_val is not None:
                    print(f"val_acc: {self.accuracy(X_val, y_val):.4f}")

    def predict(self, X):
        A_L, _ = self._forward(X)
        return get_pred(A_L)

    def accuracy(self, X, y):
        return get_accuracy(self.predict(X), y)
