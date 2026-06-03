"""
vertical_comb.py — Detection and solving of vertical-comb tasks.

A task matches VERTICAL_COMB if:
  1. Every training input has exactly one non-zero cell, on the bottom row,
     at some column C.
  2. The output draws full-height vertical stripes of that colour at columns
     C, C+2, C+4, ... (every other column rightward, stopping at grid edge).
  3. Between each consecutive pair of stripes (at C+1, C+3, ...) place a single
     colour-5 cell: even-indexed gaps (0, 2, 4, ...) get a cell at row 0 (top),
     odd-indexed gaps (1, 3, 5, ...) get a cell at the bottom row.
  4. One column past the last stripe (if within grid bounds) also gets a colour-5
     cell, continuing the same alternating pattern.

Example task: 8403a5d5
"""

VERTICAL_COMB_CATEGORIES = ["VERTICAL_COMB"]


def _apply_vertical_comb(inp: list[list[int]]) -> list[list[int]] | None:
    """Produce the vertical-comb output. Returns None if input doesn't fit."""
    H, W = len(inp), len(inp[0])

    # Find the single non-zero cell
    nonzero = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != 0]
    if len(nonzero) != 1:
        return None
    r0, C = nonzero[0]
    if r0 != H - 1:
        return None
    colour = inp[r0][C]

    out = [[0] * W for _ in range(H)]

    # Draw vertical stripes at C, C+2, C+4, ...
    stripe_cols = list(range(C, W, 2))

    for col in stripe_cols:
        for r in range(H):
            out[r][col] = colour

    # Compute gap columns and their 5-placements
    # Gaps are between consecutive stripes: C+1, C+3, ...
    # Plus one-past-last stripe if within bounds
    gap_index = 0
    for k in range(len(stripe_cols) - 1):
        gap_col = stripe_cols[k] + 1  # = C + 2k + 1
        row = 0 if gap_index % 2 == 0 else H - 1
        if 0 <= gap_col < W:
            out[row][gap_col] = 5
        gap_index += 1

    # One column past the last stripe
    extra_col = stripe_cols[-1] + 1
    if 0 <= extra_col < W:
        row = 0 if gap_index % 2 == 0 else H - 1
        out[row][extra_col] = 5

    return out


def detect_vertical_comb(task: dict) -> bool:
    """Return True if every training pair matches the vertical-comb rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        expected = _apply_vertical_comb(inp)
        if expected is None or expected != out:
            return False

    return True


def solve_vertical_comb(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the vertical-comb rule."""
    return _apply_vertical_comb(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_vertical_comb(task: dict) -> list[str]:
    """Return ['VERTICAL_COMB'] if the task matches the vertical-comb rule."""
    return VERTICAL_COMB_CATEGORIES if detect_vertical_comb(task) else []
