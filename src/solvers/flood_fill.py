"""
flood_fill.py — Rule-based solver for simple enclosed-region flood fill.

Detects the consistent fill colour from training pairs, then fills all
enclosed background cells in the test input with that colour.

Usage:
    from src.solvers.flood_fill import solve_flood_fill

    pred = solve_flood_fill(task)
    if pred is not None:
        # pred is a numpy array — the predicted test output
"""

import numpy as np
from src.categories.flood_fill import detect_flood_fill, enclosed_background


def solve_flood_fill(task: dict) -> np.ndarray | None:
    """
    Solve a task using enclosed-region flood fill.

    Reads the fill colour from training pairs, finds enclosed background
    cells in the test input, fills them, and returns the result.

    Returns a uint8 numpy array, or None if the task is not detected as
    a simple flood fill.
    """
    fill_colour = detect_flood_fill(task)
    if fill_colour is None:
        return None

    test_input = np.array(task["test"][0]["input"], dtype=np.int32)
    output = test_input.copy()
    output[enclosed_background(test_input)] = fill_colour
    return output.astype(np.uint8)
