import numpy as np


class NonSaturatingActivations:
    """
    Collection of non-saturating activation functions.

    Each activation function returns:

        out, backward

    where:
        out      : activation output
        backward : function used during backpropagation
    """

    # ============================================================
    # RELU
    # ============================================================

    @staticmethod
    def relu(x):
        """
        ReLU activation.

        f(x) = max(0, x)
        """
        x = np.asarray(x)

        mask = x > 0

        out = np.maximum(
            0.0,
            x,
        )

        def backward(dout):
            """
            Gradient of ReLU.
            """
            return dout * mask

        return out, backward

    # ============================================================
    # LEAKY RELU
    # ============================================================

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation.

        f(x) = x              if x > 0
               alpha * x      otherwise

        Parameters
        ----------
        x : np.ndarray
            Input values.

        alpha : float
            Negative slope.
        """
        x = np.asarray(x)

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        mask = x > 0

        out = np.where(
            mask,
            x,
            alpha * x,
        )

        def backward(dout):
            """
            Gradient of Leaky ReLU.
            """
            derivative = np.where(
                mask,
                1.0,
                alpha,
            )

            return dout * derivative

        return out, backward

    # ============================================================
    # ELU
    # ============================================================

    @staticmethod
    def elu(x, alpha=1.0):
        """
        Exponential Linear Unit (ELU).

        f(x) = x                          if x > 0
               alpha * (exp(x) - 1)       otherwise

        The negative input is clipped to avoid numerical overflow
        in extreme experiments.
        """
        x = np.asarray(x)

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0.")

        mask = x > 0

        # Only the negative branch requires exp().
        #
        # Clipping protects against extreme values while preserving
        # normal ELU behaviour for practical inputs.
        x_negative = np.clip(
            x,
            -80.0,
            0.0,
        )

        exp_part = alpha * (np.exp(x_negative) - 1.0)

        out = np.where(
            mask,
            x,
            exp_part,
        )

        def backward(dout):
            """
            Gradient of ELU.

            For x > 0:
                derivative = 1

            For x <= 0:
                derivative = out + alpha
            """
            derivative = np.where(
                mask,
                1.0,
                out + alpha,
            )

            return dout * derivative

        return out, backward


class SaturatingActivations:
    """
    Collection of saturating activation functions.

    These functions are especially useful in this project because
    sigmoid and tanh can demonstrate the Vanishing Gradient problem
    in deep neural networks.
    """

    # ============================================================
    # SIGMOID
    # ============================================================

    @staticmethod
    def sigmoid(x):
        """
        Numerically stable sigmoid.

        Standard formula:

            sigmoid(x) = 1 / (1 + exp(-x))

        However, directly computing exp(-x) may overflow when x is
        a very large negative number.

        Therefore:

        if x >= 0:
            sigmoid(x) = 1 / (1 + exp(-x))

        if x < 0:
            sigmoid(x) = exp(x) / (1 + exp(x))
        """
        x = np.asarray(
            x,
            dtype=float,
        )

        out = np.empty_like(
            x,
            dtype=float,
        )

        positive_mask = x >= 0
        negative_mask = ~positive_mask

        # --------------------------------------------------------
        # Positive values
        # --------------------------------------------------------
        out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))

        # --------------------------------------------------------
        # Negative values
        #
        # exp(x) is safe here because x < 0.
        # --------------------------------------------------------
        exp_x = np.exp(x[negative_mask])

        out[negative_mask] = exp_x / (1.0 + exp_x)

        def backward(dout):
            """
            Gradient of sigmoid.

            sigmoid'(x)
                = sigmoid(x) * (1 - sigmoid(x))
            """
            derivative = out * (1.0 - out)

            return dout * derivative

        return out, backward

    # ============================================================
    # TANH
    # ============================================================

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation.

        f(x) = tanh(x)
        """
        x = np.asarray(x)

        out = np.tanh(x)

        def backward(dout):
            """
            Gradient of tanh.

            tanh'(x) = 1 - tanh(x)^2
            """
            derivative = 1.0 - np.square(out)

            return dout * derivative

        return out, backward
