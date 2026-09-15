"""Shared plotting helpers.

The metrics table below replaces a matplotlib `table` block that appeared,
near-identically, five times in the original notebook.
"""

import matplotlib.pyplot as plt
import numpy as np

TABLE_FIGSIZE = [6, 4]
PLOT_FIGSIZE = [6, 4]


def metrics_table(data, columns, row_labels, title, figsize=None, decimals=4):
    """Render a small table of metrics as a matplotlib figure."""
    plt.rcParams["figure.figsize"] = figsize or TABLE_FIGSIZE
    plt.rcParams["figure.autolayout"] = True

    fig, axs = plt.subplots(1, 1)
    cell_text = np.round(np.asarray(data, dtype=float), decimals).astype(str)

    axs.axis("tight")
    axs.axis("off")
    axs.set_title(title)
    axs.table(
        cellText=cell_text, colLabels=columns, rowLabels=row_labels, loc="center"
    )
    plt.show()


def plot_curve(x, y, xlabel, ylabel, title):
    """A single labelled line plot."""
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_coefficient_path(x_values, coefficients, xlabel, feature_names=None):
    """Absolute coefficient values against a varying hyperparameter.

    The original notebook drew one figure per coefficient; overlaying them on a
    single axis makes the shrinkage ordering directly comparable.
    """
    plt.figure(figsize=PLOT_FIGSIZE)

    for j in range(coefficients.shape[1]):
        label = feature_names[j] if feature_names is not None else f"beta{j}"
        plt.plot(x_values, coefficients[:, j], label=label)

    plt.xlabel(xlabel)
    plt.ylabel("absolute coefficient value")
    plt.title(f"Effect of {xlabel} on the absolute coefficient values")
    plt.legend(fontsize="small")
    plt.show()


def plot_roc(curves):
    """ROC curves. `curves` maps a label to an (fpr, tpr) pair."""
    plt.figure(figsize=PLOT_FIGSIZE)

    for label, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


def plot_precision_recall(curves):
    """Precision-recall curves. `curves` maps a label to a (recall, precision) pair."""
    plt.figure(figsize=PLOT_FIGSIZE)

    for label, (recall, prec) in curves.items():
        plt.plot(recall, prec, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend()
    plt.show()


def plot_cosine_heatmap(matrix, tick_labels=None, title="Cosine similarity"):
    """Heatmap of a pairwise similarity matrix."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    image = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(image, ax=ax)

    if tick_labels is not None:
        ticks = np.arange(len(tick_labels))
        ax.set_xticks(ticks, tick_labels, rotation=90)
        ax.set_yticks(ticks, tick_labels)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
