"""
connect_aligned_pairs.py — Detection and solving of connect-aligned-pairs tasks.

Two or more same-coloured endpoint cells exist in the input. For every pair of
endpoint cells that share the same row or the same column, the cells strictly
between them are filled with a second (fill) colour. Endpoint cells that have
no same-row or same-column partner are left unchanged.

The endpoint colour and fill colour are inferred from the training data.

Example tasks: 253bf280, dbc1a6ce
"""

from collections import defaultdict

CONNECT_ALIGNED_PAIRS_CATEGORIES = ["CONNECT_ALIGNED_PAIRS"]


def _infer_colours(task: dict) -> tuple[int, int] | None:
    """Return (endpoint_colour, fill_colour) inferred from training pairs, or None."""
    p = task["train"][0]
    inp = p["input"]
    out = p["output"]
    if hasattr(inp[0], "tolist"):
        inp = [list(r) for r in inp]
    if hasattr(out[0], "tolist"):
        out = [list(r) for r in out]
    if not inp or not inp[0] or not out or not out[0]:
        return None
    H, W = len(inp), len(inp[0])
    Ho, Wo = len(out), len(out[0])
    in_colours = {inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0}
    out_colours = {out[r][c] for r in range(Ho) for c in range(Wo) if out[r][c] != 0}
    fill_colours = out_colours - in_colours
    if len(fill_colours) != 1 or len(in_colours) != 1:
        return None
    return next(iter(in_colours)), next(iter(fill_colours))


def _apply(inp: list[list[int]], endpoint: int, fill: int) -> list[list[int]]:
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]
    pts = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == endpoint]
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in pts:
        by_row[r].append(c)
        by_col[c].append(r)
    for r, cols in by_row.items():
        if len(cols) >= 2:
            cols.sort()
            for c in range(cols[0] + 1, cols[-1]):
                if inp[r][c] == 0:
                    out[r][c] = fill
    for c, rows in by_col.items():
        if len(rows) >= 2:
            rows.sort()
            for r in range(rows[0] + 1, rows[-1]):
                if inp[r][c] == 0:
                    out[r][c] = fill
    return out


def detect_connect_aligned_pairs(task: dict) -> bool:
    """Return True if every training pair matches the connect-aligned-pairs rule."""
    colours = _infer_colours(task)
    if colours is None:
        return False
    endpoint, fill = colours
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        if hasattr(inp[0], "tolist"):
            inp = [list(r) for r in inp]
        if hasattr(out[0], "tolist"):
            out = [list(r) for r in out]
        if not inp or not inp[0]:
            return False
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return False
        expected = _apply(inp, endpoint, fill)
        for r in range(H):
            for c in range(W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_connect_aligned_pairs(
    input_grid: list[list[int]], task: dict
) -> list[list[int]] | None:
    """Apply the connect-aligned-pairs rule using colours inferred from the task."""
    colours = _infer_colours(task)
    if colours is None:
        return None
    endpoint, fill = colours
    return _apply(input_grid, endpoint, fill)


def categorise_connect_aligned_pairs(task: dict) -> list[str]:
    return CONNECT_ALIGNED_PAIRS_CATEGORIES if detect_connect_aligned_pairs(task) else []
