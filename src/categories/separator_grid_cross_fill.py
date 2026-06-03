"""
separator_grid_cross_fill.py — Detection and solving of separator-grid cross-fill tasks.

A task matches SEPARATOR_GRID_CROSS_FILL if:
  1. Colour-8 separator lines divide the grid into a 3x3 array of cells.
  2. The 5 cross-shaped cells (top, left, centre, right, bottom) are filled with
     fixed colours: top→2, left→4, centre→6, right→3, bottom→1.
  3. The 4 corner cells remain empty (all 0).

Detection: verify this exact colour assignment holds for all training pairs.

Example task: 272f95fa
"""

SEPARATOR_GRID_CROSS_FILL_CATEGORIES = ["SEPARATOR_GRID_CROSS_FILL"]

# Fixed colour mapping: position index → colour
# Grid positions: (row, col) in the 3x3 cell array
# (0,0)=TL, (0,1)=top, (0,2)=TR  (corners stay 0)
# (1,0)=left, (1,1)=centre, (1,2)=right
# (2,0)=BL, (2,1)=bottom, (2,2)=BR  (corners stay 0)
_CROSS_COLOURS = {
    (0, 1): 2,   # top
    (1, 0): 4,   # left
    (1, 1): 6,   # centre
    (1, 2): 3,   # right
    (2, 1): 1,   # bottom
}
_CORNER_POSITIONS = {(0, 0), (0, 2), (2, 0), (2, 2)}


def _find_separator_lines(inp: list[list[int]]) -> tuple[list[int], list[int]] | None:
    """
    Find rows and columns that are entirely colour 8.
    Returns (sep_rows, sep_cols) or None if the pattern doesn't match.
    """
    H, W = len(inp), len(inp[0])
    sep_rows = [r for r in range(H) if all(inp[r][c] == 8 for c in range(W))]
    sep_cols = [c for c in range(W) if all(inp[r][c] == 8 for r in range(H))]
    return sep_rows, sep_cols


def _extract_cell_regions(
    inp: list[list[int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> list[list[tuple[int, int, int, int]]] | None:
    """
    Given separator row and column indices, extract the 3x3 grid of cell regions.
    Each cell is (row_lo, row_hi, col_lo, col_hi) — exclusive of separator lines.
    Returns None if not exactly 2 sep_rows and 2 sep_cols (which would give 3x3 cells).
    """
    if len(sep_rows) != 2 or len(sep_cols) != 2:
        return None

    H, W = len(inp), len(inp[0])
    # Row ranges: before first sep_row, between, after last sep_row
    row_ranges = [
        (0, sep_rows[0] - 1),
        (sep_rows[0] + 1, sep_rows[1] - 1),
        (sep_rows[1] + 1, H - 1),
    ]
    col_ranges = [
        (0, sep_cols[0] - 1),
        (sep_cols[0] + 1, sep_cols[1] - 1),
        (sep_cols[1] + 1, W - 1),
    ]

    # Validate non-empty ranges
    for r_lo, r_hi in row_ranges:
        if r_lo > r_hi:
            return None
    for c_lo, c_hi in col_ranges:
        if c_lo > c_hi:
            return None

    cells = []
    for ri, (r_lo, r_hi) in enumerate(row_ranges):
        row = []
        for ci, (c_lo, c_hi) in enumerate(col_ranges):
            row.append((r_lo, r_hi, c_lo, c_hi))
        cells.append(row)
    return cells


def _apply_cross_fill(inp: list[list[int]]) -> list[list[int]] | None:
    """Produce the cross-filled output. Returns None if the pattern doesn't apply."""
    H, W = len(inp), len(inp[0])
    sep_rows, sep_cols = _find_separator_lines(inp)
    cells = _extract_cell_regions(inp, sep_rows, sep_cols)
    if cells is None:
        return None

    out = [row[:] for row in inp]

    for ri in range(3):
        for ci in range(3):
            r_lo, r_hi, c_lo, c_hi = cells[ri][ci]
            fill = _CROSS_COLOURS.get((ri, ci), 0)
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    out[r][c] = fill

    return out


def detect_separator_grid_cross_fill(task: dict) -> bool:
    """Return True if every training pair matches the separator-grid cross-fill rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        expected = _apply_cross_fill(inp)
        if expected is None or expected != out:
            return False

    return True


def solve_separator_grid_cross_fill(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the cross-fill rule to a separator-grid input."""
    return _apply_cross_fill(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_separator_grid_cross_fill(task: dict) -> list[str]:
    """Return ['SEPARATOR_GRID_CROSS_FILL'] if the task matches the cross-fill rule."""
    return (
        SEPARATOR_GRID_CROSS_FILL_CATEGORIES
        if detect_separator_grid_cross_fill(task)
        else []
    )
