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
    ACTIVATION_MAP = {
                "relu": NonSaturatingActivations.relu,
                "leaky_relu": NonSaturatingActivations.leaky_relu,
                "elu": NonSaturatingActivations.elu,
                "sigmoid": SaturatingActivations.sigmoid,
                "tanh": SaturatingActivations.tanh,
                "softmax": lambda z: (softmax(z), lambda dout: dout),  # softmax + cross-entropy
                "none": lambda z: (z, lambda dout: dout),
                }

    def __init__(self, layer_dims, activations=None, seed=42,Weights_initializer=None):
        self.layer_dims = layer_dims
        self.n_layer = len(layer_dims) - 1
        self.Weights_initializer = Weights_initializer
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
        initializer = WeightInitializer(self.layer_dims, method=self.Weights_initializer, seed=self.seed)
        self.W, self.b = initializer.initialize()

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
        y_ohe = self.encoder.transform(y.reshape(-1, 1)).T
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
            learning_rate=0.1, epochs=1000, batch_size=32, print_every=10):
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

                A_L, cache = self._forward(X_batch)
                grads = self._backward(y_batch, cache)
                self._update(grads, learning_rate)

            if epoch % print_every == 0:
                A_L_full, _ = self._forward(X)
                predictions = np.argmax(A_L_full, axis=0)
                print("Epoch:", epoch)
                print("Accuracy:", accuracy_score(predictions, y))

                if X_val is not None and y_val is not None:
                    print(f"val_acc: {accuracy_score(self.predict(X_val), y_val):.4f}")

    def predict(self, X):
        A_L, _ = self._forward(X)
        return np.argmax(A_L, axis=0)
