"""
bounding_box_fill.py — Detection and solving of bounding-box fill tasks.

A task matches BOUNDING_BOX_FILL if:
  1. Every training input contains an irregular shape of colour 8 on background 0.
  2. The output fills every background (0) cell within the axis-aligned bounding box
     of the 8-shape with colour 2.
  3. The 8-cells and all cells outside the bounding box are unchanged.

Example task: 6d75e8bb
"""

BOUNDING_BOX_FILL_CATEGORIES = ["BOUNDING_BOX_FILL"]


def _apply_bounding_box_fill(inp: list[list[int]]) -> list[list[int]] | None:
    """
    Fill background cells inside the bounding box of colour-8 cells with colour 2.
    Returns None if no 8-cells are found.
    """
    H, W = len(inp), len(inp[0])

    # Find bounding box of colour-8 cells
    rows_8 = [r for r in range(H) for c in range(W) if inp[r][c] == 8]
    cols_8 = [c for r in range(H) for c in range(W) if inp[r][c] == 8]

    if not rows_8:
        return None

    min_r, max_r = min(rows_8), max(rows_8)
    min_c, max_c = min(cols_8), max(cols_8)

    out = [row[:] for row in inp]
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if out[r][c] == 0:
                out[r][c] = 2

    return out


def detect_bounding_box_fill(task: dict) -> bool:
    """Return True if every training pair matches the bounding-box fill rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        expected = _apply_bounding_box_fill(inp)
        if expected is None or expected != out:
            return False

    return True


def solve_bounding_box_fill(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Fill background cells inside the 8-shape bounding box with colour 2."""
    return _apply_bounding_box_fill(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_bounding_box_fill(task: dict) -> list[str]:
    """Return ['BOUNDING_BOX_FILL'] if the task matches the bounding-box fill rule."""
    return BOUNDING_BOX_FILL_CATEGORIES if detect_bounding_box_fill(task) else []
