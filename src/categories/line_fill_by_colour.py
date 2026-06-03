"""
line_fill_by_colour.py — Detection and solving of line-fill-by-colour tasks.

Each non-zero cell in the input fires a full line through the grid:
  - Colour 2  → vertical line  (fill the cell's entire column with 2)
  - Any other → horizontal line (fill the cell's entire row with that colour)

When a row marker and a column marker cross, the row colour wins (row fills
are applied second and overwrite the column fill at the intersection).

Example task: 178fcbfb
"""

LINE_FILL_BY_COLOUR_CATEGORIES = ["LINE_FILL_BY_COLOUR"]


def _apply_line_fill(inp: list[list[int]]) -> list[list[int]]:
    H, W = len(inp), len(inp[0])
    out = [[0] * W for _ in range(H)]
    # Pass 1: column fills (colour 2 → vertical)
    for r in range(H):
        for c in range(W):
            if inp[r][c] == 2:
                for rr in range(H):
                    out[rr][c] = 2
    # Pass 2: row fills overwrite intersections
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0 and v != 2:
                for cc in range(W):
                    out[r][cc] = v
    return out


def detect_line_fill_by_colour(task: dict) -> bool:
    """Return True if every training pair matches the line-fill-by-colour rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        if hasattr(inp[0], "tolist"):
            inp = [list(row) for row in inp]
        if hasattr(out[0], "tolist"):
            out = [list(row) for row in out]
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return False
        expected = _apply_line_fill(inp)
        for r in range(H):
            for c in range(W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_line_fill_by_colour(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the line-fill-by-colour rule."""
    H, W = len(input_grid), len(input_grid[0])
    has_marker = any(input_grid[r][c] != 0 for r in range(H) for c in range(W))
    if not has_marker:
        return None
    return _apply_line_fill(input_grid)


def categorise_line_fill_by_colour(task: dict) -> list[str]:
    return LINE_FILL_BY_COLOUR_CATEGORIES if detect_line_fill_by_colour(task) else []
