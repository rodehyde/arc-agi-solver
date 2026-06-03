"""
self_tile.py — Detection and solving of self-tiling tasks.

The 3×3 input acts as its own placement map:
  - Divide the output (9×9) into a 3×3 grid of 3×3 blocks.
  - For each cell (r, c) of the input:
      * If input[r][c] != 0 → place a full copy of the input into block (r, c).
      * If input[r][c] == 0 → fill that block with zeros.

The output is always 3× the input in each dimension (9×9 for a 3×3 input).

Example task: 007bbfb7
"""

SELF_TILE_CATEGORIES = ["SELF_TILE"]


def _apply_self_tile(inp: list[list[int]]) -> list[list[int]]:
    H, W = len(inp), len(inp[0])
    out = [[0] * (W * W) for _ in range(H * H)]
    for br in range(H):
        for bc in range(W):
            if inp[br][bc] != 0:
                for r in range(H):
                    for c in range(W):
                        out[br * H + r][bc * W + c] = inp[r][c]
    return out


def detect_self_tile(task: dict) -> bool:
    """Return True if every training pair matches the self-tile rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        if hasattr(inp[0], "tolist"):
            inp = [list(row) for row in inp]
        if hasattr(out[0], "tolist"):
            out = [list(row) for row in out]
        H, W = len(inp), len(inp[0])
        if len(out) != H * H or len(out[0]) != W * W:
            return False
        expected = _apply_self_tile(inp)
        for r in range(H * H):
            for c in range(W * W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_self_tile(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the self-tile rule."""
    return _apply_self_tile(input_grid)


def categorise_self_tile(task: dict) -> list[str]:
    return SELF_TILE_CATEGORIES if detect_self_tile(task) else []
