"""
corner_staircase.py — Detection and solving of corner-staircase expansion tasks.

A task matches CORNER_STAIRCASE if:
  1. Every training input is mostly background (0) with 2–4 non-zero cells,
     each placed at a corner of the grid (top-left, top-right, bottom-left,
     bottom-right).
  2. The output is reproduced exactly by the staircase-expansion rule below.

Rule (for a corner seed at (r0, c0) with colour X):
  Each cell (r, c) in X's Voronoi territory (strictly closest corner by
  Manhattan distance) is filled with X when:
    - dr = abs(r - r0)   [depth into grid from corner's edge]
    - dc = abs(c - c0)   [lateral distance from corner along the edge]
    - k  = dr // 2
    • (dr is even)  AND  (dc <= 2k)          → inside the solid bar, OR
    • dc >= 2*(k+1) AND  (dc % 2 == 0)       → alternating dots beyond the bar

  Cells on Voronoi boundaries (equidistant from two or more corners) are 0.

Example task: d22278a0
"""

CORNER_STAIRCASE_CATEGORIES = ["CORNER_STAIRCASE"]

_CORNERS = {
    "TL": lambda H, W: (0, 0),
    "TR": lambda H, W: (0, W - 1),
    "BL": lambda H, W: (H - 1, 0),
    "BR": lambda H, W: (H - 1, W - 1),
}


def _grid_corners(H: int, W: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}


def _staircase_fill(H: int, W: int, seeds: list[tuple[int, int, int]]) -> list[list[int]]:
    """
    Produce the staircase-expansion output grid.

    seeds: list of (row, col, colour) for each corner seed.
    """
    grid = [[0] * W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            # Voronoi: find strictly closest seed
            dists = sorted((abs(r - r0) + abs(c - c0), colour) for r0, c0, colour in seeds)
            if len(dists) >= 2 and dists[0][0] == dists[1][0]:
                continue  # tie → background

            _, colour = dists[0]
            r0, c0, _ = next(s for s in seeds if s[2] == colour and
                             abs(r - s[0]) + abs(c - s[1]) == dists[0][0])

            dr = abs(r - r0)
            dc = abs(c - c0)
            k = dr // 2

            bar_zone = (dr % 2 == 0) and (dc <= 2 * k)
            alt_zone = (dc >= 2 * (k + 1)) and (dc % 2 == 0)

            if bar_zone or alt_zone:
                grid[r][c] = colour

    return grid


def detect_corner_staircase(task: dict) -> bool:
    """
    Return True if every training pair matches the corner-staircase rule.
    """
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        corners = _grid_corners(H, W)

        # Collect non-zero input cells; all must be at corners
        seeds: list[tuple[int, int, int]] = []
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v != 0:
                    if (r, c) not in corners:
                        return False
                    seeds.append((r, c, v))

        if not (2 <= len(seeds) <= 4):
            return False

        # All seeds must have distinct colours
        if len({colour for _, _, colour in seeds}) != len(seeds):
            return False

        if _staircase_fill(H, W, seeds) != out:
            return False

    return True


def solve_corner_staircase(input_grid: list[list[int]]) -> list[list[int]] | None:
    """
    Apply the staircase-expansion rule to produce an output grid.
    Returns None if the input doesn't fit the expected form.
    """
    H, W = len(input_grid), len(input_grid[0])
    corners = _grid_corners(H, W)

    seeds: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = input_grid[r][c]
            if v != 0:
                if (r, c) not in corners:
                    return None
                seeds.append((r, c, v))

    if not (2 <= len(seeds) <= 4):
        return None

    return _staircase_fill(H, W, seeds)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_corner_staircase(task: dict) -> list[str]:
    """Return ['CORNER_STAIRCASE'] if the task matches the staircase-expansion rule."""
    return CORNER_STAIRCASE_CATEGORIES if detect_corner_staircase(task) else []
