import numpy as np

class WeightInitializer:
    def random_normal(fan_in, fan_out, scale=0.01, seed=None):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((fan_out, fan_in)) * scale

    def xavier_normal(n_in, n_out, seed=None):
        rng = np.random.default_rng(seed)
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        return rng.standard_normal((n_out, n_in)) * std_dev
    
    def he_normal(n_in, n_out, seed=None):
        rng = np.random.default_rng(seed)
        std_dev = np.sqrt(2.0 / n_in)
        return rng.standard_normal((n_out, n_in)) * std_dev

    def he_normal_leaky(n_in, n_out, negative_slope=0.01, seed=None):
        rng = np.random.default_rng(seed)
        std_dev = np.sqrt(2.0 / ((1.0 + negative_slope**2) * n_in))
        return rng.standard_normal((n_out, n_in)) * std_dev
