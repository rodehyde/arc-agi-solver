"""
tiling.py — Solvers for periodic tiling patterns.

solve_tile_fill:
  Infers the repeating tile from the non-zero cells of the test input, then
  fills every zero cell with the corresponding tile value.

solve_tile_compress:
  Extracts the minimal repeating tile from the test input.
  The tile dimensions are the smallest (oH, oW) such that
  np.tile(tile, (iH//oH, iW//oW)) == test_input.
"""

import numpy as np

from src.categories.tiling import (
    detect_tile_fill,
    detect_tile_compress,
    find_period_from_nonzero,
    reconstruct_tile,
)


def solve_tile_fill(task: dict) -> np.ndarray | None:
    """
    Solve a TILE_FILL task.

    Returns the completed grid (zeros filled with tile values), or None if
    the task is not detected as TILE_FILL or the test input cannot be solved.
    """
    if not detect_tile_fill(task):
        return None

    test_inp = np.array(task["test"][0]["input"], dtype=np.int32)

    # If the test input has no zeros, return it unchanged
    if not (test_inp == 0).any():
        return test_inp.astype(np.uint8)

    # Find the period from the non-zero cells of the test input
    period = find_period_from_nonzero(test_inp)
    if period is None:
        return None

    ph, pw = period
    tile = reconstruct_tile(test_inp, ph, pw)
    H, W = test_inp.shape
    rh = (H + ph - 1) // ph
    rw = (W + pw - 1) // pw
    tiled = np.tile(tile, (rh, rw))[:H, :W]

    # Fill only the zero cells; leave non-zero cells unchanged
    output = test_inp.copy()
    output[test_inp == 0] = tiled[test_inp == 0]
    return output.astype(np.uint8)


def solve_tile_compress(task: dict) -> np.ndarray | None:
    """
    Solve a TILE_COMPRESS task.

    Finds the minimal tile such that np.tile(tile, (rh, rw)) == test_input,
    and returns that tile.

    Returns None if not detected, or if no valid compression is found.
    """
    if not detect_tile_compress(task):
        return None

    test_inp = np.array(task["test"][0]["input"], dtype=np.int32)
    H, W = test_inp.shape

    # Find the minimal period (ph, pw) such that tiling reproduces the grid
    for ph in range(1, H + 1):
        for pw in range(1, W + 1):
            if ph == H and pw == W:
                continue  # Same size — no compression
            if H % ph != 0 or W % pw != 0:
                continue
            tile = test_inp[:ph, :pw]
            rh, rw = H // ph, W // pw
            if np.array_equal(np.tile(tile, (rh, rw)), test_inp):
                return tile.astype(np.uint8)

    return None
