import numpy as np


class WeightInitializer:
    """
    Initialize weights and biases for a fully-connected neural network.

    Parameters
    ----------
    layer_dims : list[int]
        Number of neurons in each layer.

        Example:
            [784, 64, 64, 10]

        means:
            input layer  : 784
            hidden layer : 64
            hidden layer : 64
            output layer : 10

    method : str
        Weight initialization method.

        Supported methods:
            - random_normal
            - xavier_normal
            - he_normal
            - he_normal_leaky

    seed : int | None
        Random seed used for reproducibility.
    """

    SUPPORTED_METHODS = (
        "random_normal",
        "xavier_normal",
        "he_normal",
        "he_normal_leaky",
    )

    def __init__(
        self,
        layer_dims,
        method="random_normal",
        seed=None,
    ):
        self.layer_dims = self._validate_layer_dims(layer_dims)

        self.method = self._validate_method(method)

        self.seed = seed

        # Independent random generator.
        #
        # Using default_rng avoids modifying NumPy's global
        # random state and makes experiments reproducible.
        self.rng = np.random.default_rng(seed)

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_layer_dims(layer_dims):
        """
        Validate neural-network architecture.
        """
        if not isinstance(
            layer_dims,
            (list, tuple),
        ):
            raise TypeError("layer_dims must be a list or tuple.")

        if len(layer_dims) < 2:
            raise ValueError(
                "layer_dims must contain at least "
                "an input layer and an output layer."
            )

        validated_dims = []

        for index, dim in enumerate(layer_dims):

            if not isinstance(
                dim,
                (int, np.integer),
            ):
                raise TypeError(
                    f"layer_dims[{index}] must be an integer, "
                    f"got {type(dim).__name__}."
                )

            if dim <= 0:
                raise ValueError(
                    f"layer_dims[{index}] must be greater than 0, "
                    f"got {dim}."
                )

            validated_dims.append(int(dim))

        return validated_dims

    @classmethod
    def _validate_method(cls, method):
        """
        Validate initialization method.
        """
        if not isinstance(method, str):
            raise TypeError("method must be a string.")

        if method not in cls.SUPPORTED_METHODS:
            supported = ", ".join(cls.SUPPORTED_METHODS)

            raise ValueError(
                f"Unknown initialization method '{method}'. "
                f"Supported methods: {supported}"
            )

        return method

    @staticmethod
    def _validate_fan(n_in, n_out):
        """
        Validate fan-in and fan-out values.
        """
        if not isinstance(
            n_in,
            (int, np.integer),
        ):
            raise TypeError("n_in must be an integer.")

        if not isinstance(
            n_out,
            (int, np.integer),
        ):
            raise TypeError("n_out must be an integer.")

        if n_in <= 0:
            raise ValueError("n_in must be greater than 0.")

        if n_out <= 0:
            raise ValueError("n_out must be greater than 0.")

    # ============================================================
    # RANDOM NORMAL
    # ============================================================

    def random_normal(
        self,
        n_in,
        n_out,
        scale=0.01,
    ):
        """
        Small random-normal initialization.

        W ~ N(0, scale^2)

        This method is intentionally included as a simple baseline.
        With deep networks and saturating activations such as sigmoid,
        very small weights can contribute to vanishing gradients.
        """
        self._validate_fan(
            n_in,
            n_out,
        )

        if scale <= 0:
            raise ValueError("scale must be greater than 0.")

        return self.rng.standard_normal((n_out, n_in)) * scale

    # ============================================================
    # XAVIER / GLOROT NORMAL
    # ============================================================

    def xavier_normal(
        self,
        n_in,
        n_out,
    ):
        """
        Xavier / Glorot normal initialization.

        Standard deviation:

            sqrt(2 / (fan_in + fan_out))

        Commonly suitable for activation functions such as
        sigmoid and tanh.
        """
        self._validate_fan(
            n_in,
            n_out,
        )

        std_dev = np.sqrt(2.0 / (n_in + n_out))

        return self.rng.standard_normal((n_out, n_in)) * std_dev

    # ============================================================
    # HE NORMAL
    # ============================================================

    def he_normal(
        self,
        n_in,
        n_out,
    ):
        """
        He / Kaiming normal initialization.

        Standard deviation:

            sqrt(2 / fan_in)

        Commonly paired with ReLU-family activation functions.
        """
        self._validate_fan(
            n_in,
            n_out,
        )

        std_dev = np.sqrt(2.0 / n_in)

        return self.rng.standard_normal((n_out, n_in)) * std_dev

    # ============================================================
    # HE NORMAL FOR LEAKY RELU
    # ============================================================

    def he_normal_leaky(
        self,
        n_in,
        n_out,
        negative_slope=0.01,
    ):
        """
        He initialization adapted for Leaky ReLU.

        Standard deviation:

            sqrt(
                2 /
                ((1 + negative_slope^2) * fan_in)
            )

        Parameters
        ----------
        negative_slope : float
            Negative slope used by Leaky ReLU.
        """
        self._validate_fan(
            n_in,
            n_out,
        )

        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative.")

        std_dev = np.sqrt(2.0 / ((1.0 + negative_slope**2) * n_in))

        return self.rng.standard_normal((n_out, n_in)) * std_dev

    # ============================================================
    # INITIALIZE NETWORK PARAMETERS
    # ============================================================

    def initialize(self):
        """
        Initialize all trainable parameters.

        Returns
        -------
        W : dict[int, np.ndarray]

            W[i] has shape:

                (
                    layer_dims[i],
                    layer_dims[i - 1]
                )

        b : dict[int, np.ndarray]

            b[i] has shape:

                (
                    layer_dims[i],
                    1
                )

        Layer indices run from:

            1 ... L

        where:

            L = len(layer_dims) - 1
        """

        W = {}
        b = {}

        number_of_layers = len(self.layer_dims) - 1

        # Map method names to actual functions.
        method_map = {
            "random_normal": self.random_normal,
            "xavier_normal": self.xavier_normal,
            "he_normal": self.he_normal,
            "he_normal_leaky": self.he_normal_leaky,
        }

        initializer = method_map[self.method]

        for i in range(
            1,
            number_of_layers + 1,
        ):

            fan_in = self.layer_dims[i - 1]

            fan_out = self.layer_dims[i]

            # Weight matrix
            W[i] = initializer(
                fan_in,
                fan_out,
            )

            # Bias vector
            b[i] = np.zeros(
                (
                    fan_out,
                    1,
                ),
                dtype=float,
            )

        return W, b
