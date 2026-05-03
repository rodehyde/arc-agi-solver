"""
logical_ops.py — Solver for cell-wise binary boolean operations on sub-grids.

Applies the detected split and boolean operation to the test input.
"""

import numpy as np

from src.categories.logical_ops import detect_logical_op, _try_splits, _OPS


def solve_logical_op(task: dict) -> np.ndarray | None:
    """
    Solve a logical-op task.

    Returns the predicted output grid, or None if the task is not detected
    as a logical-op task.
    """
    result = detect_logical_op(task)
    if result is None:
        return None

    split_name, op_name, fill_colour = result
    test_inp = np.array(task["test"][0]["input"], dtype=np.int32)
    H, W = test_inp.shape

    # Derive expected output dimensions from the split type
    if "vert" in split_name:
        # Vertical split: output is top or bottom half
        sep = 1 if split_name == "vert_sep" else 0
        out_h = (H - sep) // 2
        out_w = W
    else:
        # Horizontal split: output is left or right half
        sep = 1 if split_name == "horiz_sep" else 0
        out_h = H
        out_w = (W - sep) // 2

    for name, g1, g2 in _try_splits(test_inp, out_h, out_w):
        if name == split_name:
            a = g1 != 0
            b = g2 != 0
            mask = _OPS[op_name](a, b)
            output = np.zeros((out_h, out_w), dtype=np.int32)
            output[mask] = fill_colour
            return output.astype(np.uint8)

    return None
