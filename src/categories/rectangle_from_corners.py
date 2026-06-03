"""
rectangle_from_corners.py — Detection and solving of rectangle-from-corners tasks.

A task matches RECTANGLE_FROM_CORNERS if:
  1. Every training input contains only pairs of same-coloured dots on a background.
  2. The output fills the axis-aligned bounding box of each same-colour pair with
     that colour (i.e. each pair of dots marks opposite corners of a filled rectangle).
  3. Multiple colour pairs may coexist in one input.

Rule:
  For each colour C that appears exactly twice in the input, fill the rectangle
  [min_r..max_r] x [min_c..max_c] with colour C.

Example task: 56ff96f3
"""

RECTANGLE_FROM_CORNERS_CATEGORIES = ["RECTANGLE_FROM_CORNERS"]


def _filled_output(inp: list[list[int]]) -> list[list[int]]:
    """Produce the expected output: fill bounding box of each same-colour pair."""
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]

    # Collect positions of each colour
    from collections import defaultdict
    positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if inp[r][c] != 0:
                positions[inp[r][c]].append((r, c))

    # For each colour that appears exactly twice, fill the bounding box
    for colour, pts in positions.items():
        if len(pts) != 2:
            continue
        r0, c0 = pts[0]
        r1, c1 = pts[1]
        min_r, max_r = min(r0, r1), max(r0, r1)
        min_c, max_c = min(c0, c1), max(c0, c1)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                out[r][c] = colour

    return out


def detect_rectangle_from_corners(task: dict) -> bool:
    """Return True if every training pair matches the rectangle-from-corners rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        # All non-zero colours must appear exactly twice in input
        from collections import defaultdict
        positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    positions[inp[r][c]].append((r, c))

        if not positions:
            return False

        for colour, pts in positions.items():
            if len(pts) != 2:
                return False

        # Verify the output matches our rule
        expected = _filled_output(inp)
        if expected != out:
            return False

    return True


def solve_rectangle_from_corners(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Fill bounding box of each same-colour dot pair with that colour."""
    from collections import defaultdict
    H, W = len(input_grid), len(input_grid[0])

    positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] != 0:
                positions[input_grid[r][c]].append((r, c))

    if not positions:
        return None

    for pts in positions.values():
        if len(pts) != 2:
            return None

    return _filled_output(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_rectangle_from_corners(task: dict) -> list[str]:
    """Return ['RECTANGLE_FROM_CORNERS'] if the task matches the rule."""
    return RECTANGLE_FROM_CORNERS_CATEGORIES if detect_rectangle_from_corners(task) else []
