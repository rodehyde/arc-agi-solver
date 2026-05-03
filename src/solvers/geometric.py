"""
geometric.py — Rule-based solver for whole-grid geometric transforms.

Works for any task where a single deterministic function (flip, rotate,
mirror-append, quadrant-tile) maps every training input exactly to its
training output.  The same function is applied to the test input.

Usage:
    from src.solvers.geometric import solve_geometric

    pred = solve_geometric(task)   # task dict with 'train' and 'test' keys
    if pred is not None:
        # pred is a numpy array — the predicted test output
"""

import numpy as np
from src.categories.geometric_transforms import detect_transform


def solve_geometric(task: dict) -> np.ndarray | None:
    """
    Solve a task using whole-grid geometric transform detection.

    Detects the unique transform that maps every training input to its
    training output, then applies it to the first test input.

    Returns the predicted output as a uint8 numpy array, or None if no
    consistent transform is found.
    """
    result = detect_transform(task)
    if result is None:
        return None
    _, fn = result
    test_input = np.array(task["test"][0]["input"], dtype=np.int32)
    return fn(test_input).astype(np.uint8)
