"""
logical_ops.py — Detection of cell-wise binary boolean operations on sub-grids.

A task passes if:
  1. The input splits into two equal sub-grids matching the output shape.
     Split is vertical (top/bottom) or horizontal (left/right),
     with or without a single separator row/column.
  2. Converting both sub-grids to boolean (non-zero = True) and applying one
     of 8 binary operations produces a mask consistent with the output:
         output = fill_colour where mask is True, 0 elsewhere.
  3. The fill colour is a single consistent value across all training pairs.

Operations tried: AND, OR, XOR, NOR, NAND, XNOR, A_NOT_B, B_NOT_A.
"""

import numpy as np

LOGICAL_OP_CATEGORIES = ["LOGICAL_OP"]

_OPS: dict[str, object] = {
    "AND":     lambda a, b: a & b,
    "OR":      lambda a, b: a | b,
    "XOR":     lambda a, b: a ^ b,
    "NOR":     lambda a, b: ~(a | b),
    "NAND":    lambda a, b: ~(a & b),
    "XNOR":    lambda a, b: ~(a ^ b),
    "A_NOT_B": lambda a, b: a & ~b,
    "B_NOT_A": lambda a, b: ~a & b,
}


def _try_splits(
    inp: np.ndarray, out_h: int, out_w: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return all geometrically valid (split_name, g1, g2) for this input."""
    H, W = inp.shape
    results = []
    if H == 2 * out_h and W == out_w:
        results.append(("vert",      inp[:out_h],       inp[out_h:]))
    if H == 2 * out_h + 1 and W == out_w:
        results.append(("vert_sep",  inp[:out_h],       inp[out_h + 1:]))
    if H == out_h and W == 2 * out_w:
        results.append(("horiz",     inp[:, :out_w],    inp[:, out_w:]))
    if H == out_h and W == 2 * out_w + 1:
        results.append(("horiz_sep", inp[:, :out_w],    inp[:, out_w + 1:]))
    return results


def detect_logical_op(task: dict) -> tuple[str, str, int] | None:
    """
    Detect a cell-wise binary boolean operation on two sub-grids.

    Returns (split_name, op_name, fill_colour), or None.
    """
    pairs = task["train"]
    if not pairs:
        return None

    # Output shape must be the same across all pairs
    out_shapes = {(len(p["output"]), len(p["output"][0])) for p in pairs}
    if len(out_shapes) != 1:
        return None
    out_h, out_w = next(iter(out_shapes))

    candidates: set | None = None

    for p in pairs:
        inp = np.array(p["input"],  dtype=np.int32)
        out = np.array(p["output"], dtype=np.int32)

        # Output must be binary: exactly one non-zero colour
        out_vals = set(int(v) for v in out.flatten() if v != 0)
        if len(out_vals) != 1:
            return None
        fill_colour = next(iter(out_vals))

        if not np.all((out == 0) | (out == fill_colour)):
            return None

        expected_mask = out == fill_colour

        # Find all (split, op, fill) consistent with this pair
        pair_hits: set[tuple[str, str, int]] = set()
        for split_name, g1, g2 in _try_splits(inp, out_h, out_w):
            a = g1 != 0
            b = g2 != 0
            for op_name, op_fn in _OPS.items():
                if np.array_equal(op_fn(a, b), expected_mask):
                    pair_hits.add((split_name, op_name, fill_colour))

        if not pair_hits:
            return None

        candidates = pair_hits if candidates is None else candidates & pair_hits
        if not candidates:
            return None

    if not candidates:
        return None

    # Prefer deterministic ordering: sort and return first
    return sorted(candidates)[0]


def solve_logical_op(
    input_grid: list[list[int]], task: dict
) -> list[list[int]] | None:
    """Apply the detected logical operation to input_grid."""
    result = detect_logical_op(task)
    if result is None:
        return None
    split_name, op_name, fill_colour = result
    inp = np.array(input_grid, dtype=np.int32)
    out_h = len(task["train"][0]["output"])
    out_w = len(task["train"][0]["output"][0])
    splits = _try_splits(inp, out_h, out_w)
    split_map = {s[0]: (s[1], s[2]) for s in splits}
    if split_name not in split_map:
        return None
    g1, g2 = split_map[split_name]
    a, b = g1 != 0, g2 != 0
    mask = _OPS[op_name](a, b)
    out = np.where(mask, fill_colour, 0).astype(np.int32)
    return out.tolist()


def categorise_logical_op(task: dict) -> list[str]:
    """Return ['LOGICAL_OP'] if a consistent binary boolean operation is detected."""
    return LOGICAL_OP_CATEGORIES if detect_logical_op(task) is not None else []
