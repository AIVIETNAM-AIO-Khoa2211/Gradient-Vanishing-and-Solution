"""Comparison plots for vanishing/exploding-gradient experiments.

The public plotting functions receive a mapping of experiment names to
histories, for example::

    histories = {
        "Baseline": baseline_history,
        "He + LeakyReLU": he_history,
        "BatchNorm": batchnorm_history,
    }

Each history must contain ``epoch`` and the metrics used by the requested
plot. Layer-wise metrics such as ``dW_rms`` have shape
``(n_records, n_layers)``. RMS is preferred to a raw norm because it is
comparable across layers with different numbers of parameters.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12
_ARRAY_KEYS = (
    "epoch",
    "loss",  # Legacy alias for train_loss.
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "dW_norm",
    "dW_rms",
    "dZ_norm",
    "dZ_rms",
)


# --------------------------------------------------------------------------- IO
def save_history(history, path):
    """Save supported numeric history arrays to an ``.npz`` file."""
    validate_history(history)
    arrays = {
        key: np.asarray(history[key])
        for key in _ARRAY_KEYS
        if key in history
    }
    np.savez(path, **arrays)


def load_history(path):
    """Load and validate a history saved by :func:`save_history`."""
    with np.load(path, allow_pickle=False) as file:
        history = {key: file[key] for key in file.files}
    validate_history(history)
    return history


# ------------------------------------------------------------------- validation
def validate_history(history, required_keys=()):
    """Validate one history and raise a clear error for invalid data."""
    if not isinstance(history, Mapping):
        raise TypeError("history must be a mapping of metric names to arrays")
    if "epoch" not in history:
        raise KeyError("history is missing required key 'epoch'")

    missing = [key for key in required_keys if key not in history]
    if missing:
        raise KeyError(f"history is missing required keys: {missing}")

    epochs = np.asarray(history["epoch"])
    if epochs.ndim != 1 or epochs.size == 0:
        raise ValueError("history['epoch'] must be a non-empty 1-D array")
    if not np.issubdtype(epochs.dtype, np.number):
        raise TypeError("history['epoch'] must contain numeric values")
    if not np.all(np.isfinite(epochs)):
        raise ValueError("history['epoch'] contains NaN or infinity")
    if epochs.size > 1 and np.any(np.diff(epochs) <= 0):
        raise ValueError("history['epoch'] must be strictly increasing")

    for key in required_keys:
        values = np.asarray(history[key])
        if values.ndim == 0 or values.shape[0] != epochs.size:
            raise ValueError(
                f"history['{key}'] must contain {epochs.size} records"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"history['{key}'] must contain numeric values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"history['{key}'] contains NaN or infinity")


def _validate_histories(histories):
    if not isinstance(histories, Mapping) or not histories:
        raise ValueError(
            "histories must be a non-empty mapping: "
            "{'pipeline name': history, ...}"
        )
    for name, history in histories.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each pipeline must have a non-empty string name")
        try:
            validate_history(history)
        except (TypeError, ValueError, KeyError) as error:
            raise type(error)(f"invalid history for pipeline '{name}': {error}") from error


def _layer_metric(history, key):
    validate_history(history, required_keys=(key,))
    data = np.asarray(history[key], dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError(
            f"history['{key}'] must have shape (n_records, n_layers)"
        )
    if np.any(data < 0):
        raise ValueError(f"history['{key}'] must contain non-negative values")
    return data


def _scalar_metric(history, key, *, nonnegative=False):
    validate_history(history, required_keys=(key,))
    values = np.asarray(history[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"history['{key}'] must be a 1-D array")
    if nonnegative and np.any(values < 0):
        raise ValueError(f"history['{key}'] must contain non-negative values")
    return values


def _metric_key(history, metric, split):
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")
    key = f"{split}_{metric}"
    if metric == "loss" and split == "train" and key not in history:
        key = "loss"  # Compatibility with the original history schema.
    return key


def _get_ax(ax, figsize=(7, 4.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _record_index(epochs, epoch):
    """Resolve ``-1``/``None`` as final record, otherwise nearest epoch."""
    if epoch is None or epoch == -1:
        return len(epochs) - 1
    if not np.isscalar(epoch) or not np.isfinite(epoch):
        raise ValueError("epoch must be a finite number, -1, or None")
    return int(np.argmin(np.abs(epochs - epoch)))


# ---------------------------------------------------------------------- metrics
def gradient_ratio(history, key="dW_rms"):
    """Return layer-1 / last-layer gradient magnitude for each record."""
    data = _layer_metric(history, key)
    return data[:, 0] / np.maximum(data[:, -1], EPS)


# ------------------------------------------------------------------------ plots
def plot_grad_by_layer(histories, epoch=-1, key="dW_rms", ax=None):
    """Compare gradient magnitude across layers at one training time.

    ``epoch=-1`` selects the last available record of each pipeline. For a
    numeric epoch, the closest recorded epoch is used and shown in the legend.
    """
    _validate_histories(histories)
    fig, ax = _get_ax(ax)

    for name, history in histories.items():
        epochs = np.asarray(history["epoch"], dtype=float)
        data = _layer_metric(history, key)
        index = _record_index(epochs, epoch)
        layers = np.arange(1, data.shape[1] + 1)
        ax.plot(
            layers,
            np.maximum(data[index], EPS),
            marker="o",
            ms=4,
            label=f"{name} (epoch {epochs[index]:g})",
        )

    ax.set(
        yscale="log",
        xlabel="Layer",
        ylabel=key,
        title=f"Gradient by layer ({key})",
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return fig, ax


def plot_grad_over_epochs(
    histories,
    layer=1,
    key="dW_rms",
    metric="layer",
    ax=None,
):
    """Compare gradient evolution of multiple pipelines.

    With ``metric='layer'``, plot the selected layer (layer 1 by default).
    With ``metric='ratio'``, plot layer-1 / last-layer gradient magnitude.
    Averaging all layers is intentionally avoided because it can hide a
    vanishing gradient in the early layers.
    """
    _validate_histories(histories)
    if metric not in {"layer", "ratio"}:
        raise ValueError("metric must be either 'layer' or 'ratio'")
    if not isinstance(layer, (int, np.integer)):
        raise TypeError("layer must be an integer")

    fig, ax = _get_ax(ax)
    for name, history in histories.items():
        epochs = np.asarray(history["epoch"], dtype=float)
        data = _layer_metric(history, key)
        if metric == "ratio":
            values = gradient_ratio(history, key)
        else:
            if layer < 1 or layer > data.shape[1]:
                raise ValueError(
                    f"layer {layer} is invalid for pipeline '{name}'; "
                    f"expected 1..{data.shape[1]}"
                )
            values = data[:, layer - 1]
        ax.plot(epochs, np.maximum(values, EPS), label=name)

    if metric == "ratio":
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ylabel = f"{key}: layer 1 / last layer"
        title = "Gradient ratio over epochs"
    else:
        ylabel = key
        title = f"Gradient at layer {layer} over epochs"

    ax.set(yscale="log", xlabel="Epoch", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    return fig, ax


def plot_loss_over_epochs(histories, split="val", ax=None):
    """Compare train or validation loss across pipelines."""
    _validate_histories(histories)
    fig, ax = _get_ax(ax)

    for name, history in histories.items():
        key = _metric_key(history, "loss", split)
        epochs = np.asarray(history["epoch"], dtype=float)
        values = _scalar_metric(history, key, nonnegative=True)
        ax.plot(epochs, values, label=name)

    ax.set(
        xlabel="Epoch",
        ylabel=f"{split.capitalize()} loss",
        title=f"{split.capitalize()} loss over epochs",
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_acc_over_epochs(histories, split="val", ax=None):
    """Compare train or validation accuracy across pipelines."""
    _validate_histories(histories)
    fig, ax = _get_ax(ax)

    for name, history in histories.items():
        key = _metric_key(history, "acc", split)
        epochs = np.asarray(history["epoch"], dtype=float)
        values = _scalar_metric(history, key)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(
                f"history['{key}'] for pipeline '{name}' must be between 0 and 1"
            )
        ax.plot(epochs, values, label=name)

    ax.set(
        xlabel="Epoch",
        ylabel=f"{split.capitalize()} accuracy",
        title=f"{split.capitalize()} accuracy over epochs",
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_experiment_summary(
    histories,
    epoch=-1,
    layer=1,
    key="dW_rms",
    split="val",
):
    """Create the four report plots in one 2x2 dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_grad_by_layer(histories, epoch=epoch, key=key, ax=axes[0, 0])
    plot_norm_over_epochs(
        histories,
        layer=layer,
        key=key,
        metric="layer",
        ax=axes[0, 1],
    )
    plot_loss_over_epochs(histories, split=split, ax=axes[1, 0])
    plot_acc_over_epochs(histories, split=split, ax=axes[1, 1])
    fig.tight_layout()
    return fig, axes
