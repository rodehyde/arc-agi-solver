"""
visualise.py — Plotting utilities for ARC-AGI grids and tasks.

Standard ARC colour palette (indices 0–9):
    0 black, 1 blue, 2 red, 3 green, 4 yellow,
    5 grey, 6 magenta, 7 orange, 8 azure, 9 maroon
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

# Official ARC colour map
ARC_COLOURS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]

ARC_CMAP = ListedColormap(ARC_COLOURS)


def plot_grid(ax, grid: list[list[int]], title: str = "") -> None:
    """Draw a single ARC grid on a matplotlib Axes."""
    arr = np.array(grid)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")

    # Grid lines
    rows, cols = arr.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    if title:
        ax.set_title(title, fontsize=9)


def plot_task(task: dict, max_pairs: int = 4) -> plt.Figure:
    """
    Plot all train pairs (input → output) for a task, plus the test input.

    Returns the Figure so you can call fig.savefig(...) or plt.show().
    """
    train_pairs = task["train"][:max_pairs]
    test_inputs = [ex["input"] for ex in task.get("test", [])]
    n_train = len(train_pairs)
    n_test = len(test_inputs)
    n_cols = n_train + n_test
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    # Make axes always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    task_id = task.get("task_id", "unknown")
    fig.suptitle(f"Task: {task_id}", fontsize=11, fontweight="bold")

    for i, pair in enumerate(train_pairs):
        plot_grid(axes[0, i], pair["input"],  title=f"Train {i+1} — Input")
        plot_grid(axes[1, i], pair["output"], title=f"Train {i+1} — Output")

    for j, inp in enumerate(test_inputs):
        col = n_train + j
        plot_grid(axes[0, col], inp, title=f"Test {j+1} — Input")
        axes[1, col].axis("off")
        axes[1, col].text(0.5, 0.5, "?", ha="center", va="center",
                          fontsize=40, color="grey",
                          transform=axes[1, col].transAxes)

    fig.tight_layout()
    return fig


def plot_category_sample(tasks: list[dict], category: str,
                         n: int = 3) -> plt.Figure:
    """
    Plot the first train pair from each of n tasks in a given category.
    Useful for quickly eyeballing what a category looks like.
    """
    sample = tasks[:n]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Category: {category}  ({n} examples)", fontsize=12,
                 fontweight="bold")

    for i, task in enumerate(sample):
        pair = task["train"][0]
        task_id = task.get("task_id", "")
        plot_grid(axes[0, i], pair["input"],  title=f"{task_id}\nInput")
        plot_grid(axes[1, i], pair["output"], title="Output")

    fig.tight_layout()
    return fig
