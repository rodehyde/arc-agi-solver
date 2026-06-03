"""
separator_grid_diagonal_fill.py — Detection and solving of separator-grid diagonal-fill.

A task matches SEPARATOR_GRID_DIAGONAL_FILL if:
  1. Colour-5 separator lines divide the grid into an NxM array of cells
     (variable dimensions).
  2. The top-left cell is filled with colour 1.
  3. The centre cell (row_count//2, col_count//2) is filled with colour 2.
  4. The bottom-right cell is filled with colour 3.
  5. All other cells remain empty (all 0).

Example task: 941d9a10
"""

SEPARATOR_GRID_DIAGONAL_FILL_CATEGORIES = ["SEPARATOR_GRID_DIAGONAL_FILL"]


def _find_full_separator_lines(
    inp: list[list[int]], sep_colour: int = 5
) -> tuple[list[int], list[int]]:
    """
    Return (sep_rows, sep_cols): indices of rows/columns entirely of sep_colour.
    """
    H, W = len(inp), len(inp[0])
    sep_rows = [r for r in range(H) if all(inp[r][c] == sep_colour for c in range(W))]
    sep_cols = [c for c in range(W) if all(inp[r][c] == sep_colour for r in range(H))]
    return sep_rows, sep_cols


def _cell_regions(
    H: int, W: int, sep_rows: list[int], sep_cols: list[int]
) -> list[list[tuple[int, int, int, int]]]:
    """
    Given separator rows and columns, return the list-of-lists of cell bounding boxes.
    Each cell: (r_lo, r_hi, c_lo, c_hi).
    """
    all_row_seps = sorted(sep_rows)
    all_col_seps = sorted(sep_cols)

    row_ranges = []
    prev = 0
    for sr in all_row_seps:
        if sr > prev:
            row_ranges.append((prev, sr - 1))
        prev = sr + 1
    if prev <= H - 1:
        row_ranges.append((prev, H - 1))

    col_ranges = []
    prev = 0
    for sc in all_col_seps:
        if sc > prev:
            col_ranges.append((prev, sc - 1))
        prev = sc + 1
    if prev <= W - 1:
        col_ranges.append((prev, W - 1))

    cells = []
    for r_lo, r_hi in row_ranges:
        row = []
        for c_lo, c_hi in col_ranges:
            row.append((r_lo, r_hi, c_lo, c_hi))
        cells.append(row)
    return cells


def _apply_diagonal_fill(inp: list[list[int]]) -> list[list[int]] | None:
    """Produce the diagonal-fill output. Returns None if pattern doesn't apply."""
    H, W = len(inp), len(inp[0])
    sep_rows, sep_cols = _find_full_separator_lines(inp)

    if not sep_rows or not sep_cols:
        return None

    cells = _cell_regions(H, W, sep_rows, sep_cols)
    n_rows = len(cells)
    n_cols = len(cells[0]) if cells else 0

    if n_rows < 1 or n_cols < 1:
        return None

    # Fill positions
    fill_map: dict[tuple[int, int], int] = {}
    fill_map[(0, 0)] = 1                                   # top-left
    fill_map[(n_rows // 2, n_cols // 2)] = 2               # centre
    fill_map[(n_rows - 1, n_cols - 1)] = 3                 # bottom-right

    out = [row[:] for row in inp]

    for ri, row in enumerate(cells):
        for ci, (r_lo, r_hi, c_lo, c_hi) in enumerate(row):
            fill = fill_map.get((ri, ci), 0)
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    out[r][c] = fill

    return out


def detect_separator_grid_diagonal_fill(task: dict) -> bool:
    """Return True if every training pair matches the diagonal-fill rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        expected = _apply_diagonal_fill(inp)
        if expected is None or expected != out:
            return False

    return True


def solve_separator_grid_diagonal_fill(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the separator-grid diagonal-fill rule."""
    return _apply_diagonal_fill(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_separator_grid_diagonal_fill(task: dict) -> list[str]:
    """Return ['SEPARATOR_GRID_DIAGONAL_FILL'] if the task matches the rule."""
    return (
        SEPARATOR_GRID_DIAGONAL_FILL_CATEGORIES
        if detect_separator_grid_diagonal_fill(task)
        else []
    )
