"""
quadrant_mirror.py — Detection and solving of quadrant-mirror tasks.

The input is an H×W grid. The output is a 2H×2W grid formed by placing the
input in all four quadrants with mirroring:
  top-left     = original
  top-right    = flip_h (mirror left-right)
  bottom-left  = flip_v (mirror top-bottom)
  bottom-right = rot180 (flip both)

The result has full bilateral symmetry about both the horizontal and vertical
centre axes.

Example tasks: 3af2c5a8, 62c24649, 67e8384a
"""

QUADRANT_MIRROR_CATEGORIES = ["QUADRANT_MIRROR"]


def _apply(inp: list[list[int]]) -> list[list[int]]:
    H, W = len(inp), len(inp[0])
    out = [[0] * (2 * W) for _ in range(2 * H)]
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            out[r][c] = v
            out[r][2 * W - 1 - c] = v
            out[2 * H - 1 - r][c] = v
            out[2 * H - 1 - r][2 * W - 1 - c] = v
    return out


def detect_quadrant_mirror(task: dict) -> bool:
    """Return True if every training pair matches the quadrant-mirror rule."""
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
        if len(out) != 2 * H or len(out[0]) != 2 * W:
            return False
        expected = _apply(inp)
        for r in range(2 * H):
            for c in range(2 * W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_quadrant_mirror(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the quadrant-mirror rule."""
    if not input_grid or not input_grid[0]:
        return None
    return _apply(input_grid)


def categorise_quadrant_mirror(task: dict) -> list[str]:
    return QUADRANT_MIRROR_CATEGORIES if detect_quadrant_mirror(task) else []
