import numpy as np

class WeightInitializer:
    def __init__(self, layer_dims, method="random_normal", seed = None ):
        """
        layer_dims : list, e.g. [784, 128, 64, 10]
        input size, hidden sizes..., output size.
        """
        self.method = method
        self.seed = seed
        self.layer_dims = layer_dims
        self.rng = np.random.default_rng(seed)

    def random_normal(self, n_in, n_out, scale=0.1):
        return self.rng.standard_normal((n_out, n_in)) * scale

    def xavier_normal(self, n_in, n_out):
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        return self.rng.standard_normal((n_out, n_in)) * std_dev
    
    def he_normal(self, n_in, n_out):
        std_dev = np.sqrt(2.0 / n_in)
        return self.rng.standard_normal((n_out, n_in)) * std_dev

    def he_normal_leaky(self, n_in, n_out, negative_slope=0.01):
        std_dev = np.sqrt(2.0 / ((1.0 + negative_slope**2) * n_in))
        return self.rng.standard_normal((n_out, n_in)) * std_dev
    def initialize(self) -> tuple[dict, dict]:
        """
        Build the parameters for every weighted layer, using self.method.

        Returns (W, b):
        W : dict {i: array of shape (layer_dims[i], layer_dims[i-1])}, i = 1..L
        b : dict {i: np.zeros((layer_dims[i], 1))}, same keys as W
        (L = len(layer_dims) - 1; there is no key 0.)

        For layer i: fan_in = layer_dims[i-1], fan_out = layer_dims[i].
        Loop i = 1..L in order, look up the method by name (self.method), and
        call it with (fan_in, fan_out) so that all layers draw from the single
        self.rng. Raise ValueError if self.method is not a valid name.
        """
        W = {}
        b = {}
        L = len(self.layer_dims) - 1  
        method_map = {
            "random_normal": self.random_normal,
            "xavier_normal": self.xavier_normal,
            "he_normal": self.he_normal,
            "he_normal_leaky": self.he_normal_leaky
        }
        if self.method not in method_map:
            raise ValueError(f"Method khởi tạo '{self.method}' không tồn tại!")
            
        for i in range(1, L + 1):
            n_in = self.layer_dims[i-1]
            n_out = self.layer_dims[i]
            
            W[i] = method_map[self.method](n_in, n_out)
            b[i] = np.zeros((n_out, 1))
            
        return W, b