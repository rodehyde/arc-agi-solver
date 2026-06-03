"""
row_fill_meet_middle.py — Detection and solving of row-fill-meet-middle tasks.

For each row that has non-zero values at both its left edge (col 0) and its
right edge (last column):
  - Fill the left half of the row with the left colour.
  - Place colour 5 at the exact centre column (width // 2).
  - Fill the right half with the right colour.

Rows without values at both edges are left unchanged.

Example task: 29c11459
"""

ROW_FILL_MEET_MIDDLE_CATEGORIES = ["ROW_FILL_MEET_MIDDLE"]


def _apply_row_fill(inp: list[list[int]]) -> list[list[int]]:
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]
    mid = W // 2
    for r in range(H):
        left_v = inp[r][0]
        right_v = inp[r][W - 1]
        if left_v != 0 and right_v != 0:
            for c in range(mid):
                out[r][c] = left_v
            out[r][mid] = 5
            for c in range(mid + 1, W):
                out[r][c] = right_v
    return out


def detect_row_fill_meet_middle(task: dict) -> bool:
    """Return True if every training pair matches the row-fill-meet-middle rule."""
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
        expected = _apply_row_fill(inp)
        for r in range(H):
            for c in range(W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_row_fill_meet_middle(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the row-fill-meet-middle rule."""
    H, W = len(input_grid), len(input_grid[0])
    has_pair = any(
        input_grid[r][0] != 0 and input_grid[r][W - 1] != 0
        for r in range(H)
    )
    if not has_pair:
        return None
    return _apply_row_fill(input_grid)


def categorise_row_fill_meet_middle(task: dict) -> list[str]:
    return ROW_FILL_MEET_MIDDLE_CATEGORIES if detect_row_fill_meet_middle(task) else []
