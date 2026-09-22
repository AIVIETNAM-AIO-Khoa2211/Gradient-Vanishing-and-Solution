"""Gradient diagnostics for vanishing / exploding gradients.

All functions take the `history` dict returned by NeuralNetwork.fit()
(arrays of shape [n_records, n_layer], column j = layer j + 1) and
return (fig, ax) so the caller decides whether to show or save.
"""
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12
_ARRAY_KEYS = ("epoch", "loss", "train_acc", "dW_norm", "dW_rms", "dZ_norm")


# ---------------------------------------------------------------- io
def save_history(history, path):
    """Save scalar history to .npz (raw snapshots are not included)."""
    np.savez(path, **{k: np.asarray(history[k]) for k in _ARRAY_KEYS if k in history})

def load_history(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------- metrics
def gradient_ratio(history, key="dW_norm"):
    """norm(layer 1) / norm(layer L) per record. ~1 ok, <<1 vanish, >>1 explode."""
    data = history[key]
    return data[:, 0] / (data[:, -1] + EPS)

def decay_slope(history, key="dW_norm"):
    """Slope of log10(norm) vs layer index, per record.
    > 0: norm shrinks toward early layers (vanish); < 0: explode; ~0: stable.
    Unit: orders of magnitude per layer."""
    y = np.log10(history[key].T + EPS)                  # [L, n_records]
    x = np.arange(1, y.shape[0] + 1, dtype=float)
    xc = x - x.mean()
    return (xc[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()


# ---------------------------------------------------------------- helpers
def _get_ax(ax, figsize=(7, 4.5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax

def _default_layers(n_layer, k=5):
    return sorted({int(l) for l in np.linspace(1, n_layer, k)})


# ---------------------------------------------------------------- plots
def plot_norm_by_layer(history, key="dW_norm", fractions=(0, 0.1, 0.5, 1.0), ax=None):
    """Plot 1: norm vs layer (log y) at several points of training."""
    fig, ax = _get_ax(ax)
    epochs, data = history["epoch"], history[key]
    layers = np.arange(1, data.shape[1] + 1)

    idxs = sorted({int(np.argmin(np.abs(epochs - f * epochs[-1]))) for f in fractions})
    for idx in idxs:
        ax.plot(layers, data[idx] + EPS, marker="o", ms=3, label=f"epoch {epochs[idx]}")

    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel(key)
    ax.set_title(f"{key} by layer")
    ax.legend()
    return fig, ax

def plot_heatmap(history, key="dW_norm", ax=None):
    """Plot 2: x = epoch, y = layer, colour = log10(norm)."""
    fig, ax = _get_ax(ax, figsize=(8, 4.5))
    epochs, data = history["epoch"], history[key]
    L = data.shape[1]

    img = ax.imshow(np.log10(data.T + EPS), aspect="auto", origin="lower",
                    extent=[epochs[0], epochs[-1], 0.5, L + 0.5], cmap="viridis")
    fig.colorbar(img, ax=ax, label=f"log10({key})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("layer")
    ax.set_title(f"{key} heatmap")
    return fig, ax

def plot_norm_over_epochs(history, layers=None, key="dW_norm", ax=None):
    """Plot 3: norm vs epoch for a few representative layers (log y)."""
    fig, ax = _get_ax(ax)
    epochs, data = history["epoch"], history[key]
    layers = layers or _default_layers(data.shape[1])

    for l in layers:
        ax.plot(epochs, data[:, l - 1] + EPS, label=f"layer {l}")

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.set_title(f"{key} over training")
    ax.legend()
    return fig, ax

def plot_grad_histograms(snapshots, epoch, layers=None, bins=60, ax=None):
    """Plot 4: distribution of log10|dW| for a few layers at one snapshot epoch.
    The legend shows the fraction of exactly-zero gradients (dead ReLU units)."""
    fig, ax = _get_ax(ax)
    snap = snapshots[epoch]
    layers = layers or _default_layers(len(snap), k=4)

    logs = {l: np.log10(snap[l][snap[l] > 0]) for l in layers}
    nonempty = [v for v in logs.values() if v.size]
    edges = None
    if nonempty:
        lo, hi = min(v.min() for v in nonempty), max(v.max() for v in nonempty)
        edges = np.linspace(lo, hi + 1e-9, bins + 1)

    for l in layers:
        zero_frac = np.mean(snap[l] == 0)
        label = f"layer {l} (zeros {zero_frac:.0%})"
        if logs[l].size:
            ax.hist(logs[l], bins=edges, histtype="step", label=label)
        else:
            ax.plot([], [], label=label)

    ax.set_xlabel("log10 |dW|")
    ax.set_ylabel("count")
    ax.set_title(f"|dW| distribution (epoch {epoch})")
    ax.legend()
    return fig, ax

def plot_diagnostics(history, snapshots=None, key="dW_norm"):
    """2x2 overview of the four plots above."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_norm_by_layer(history, key, ax=axes[0, 0])
    plot_heatmap(history, key, ax=axes[0, 1])
    plot_norm_over_epochs(history, key=key, ax=axes[1, 0])
    if snapshots:
        plot_grad_histograms(snapshots, min(snapshots), ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------- comparison
def compare_profiles(histories, record=0, key="dW_norm", ax=None):
    """Norm vs layer for several configs at the same record index
    (0 = initialisation, -1 = end of training).
    histories: {"baseline": hist1, "he": hist2, ...}"""
    fig, ax = _get_ax(ax)
    for name, h in histories.items():
        data = h[key]
        ax.plot(np.arange(1, data.shape[1] + 1), data[record] + EPS,
                marker="o", ms=3, label=f"{name} (epoch {h['epoch'][record]})")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel(key)
    ax.set_title(f"{key} by layer: comparison")
    ax.legend()
    return fig, ax

def compare_ratio(histories, key="dW_norm", ax=None):
    """norm(layer 1) / norm(layer L) over epochs for several configs."""
    fig, ax = _get_ax(ax)
    for name, h in histories.items():
        ax.plot(h["epoch"], gradient_ratio(h, key) + EPS, label=name)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("norm(layer 1) / norm(layer L)")
    ax.set_title("Gradient ratio: comparison")
    ax.legend()
    return fig, ax