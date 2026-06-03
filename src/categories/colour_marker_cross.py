"""
colour_marker_cross.py — Detection and solving of colour-marker cross tasks.

A task matches COLOUR_MARKER_CROSS if:
  1. Every training input contains some colour-1 cells and/or colour-2 cells
     (plus possibly other unchanged cells).
  2. For each colour-1 cell, its 4 orthogonal neighbours are painted colour 7 (+cross).
  3. For each colour-2 cell, its 4 diagonal neighbours are painted colour 4 (×cross).
  4. All other cells remain unchanged. Clips to grid bounds.

Example task: 0ca9ddb6
"""

COLOUR_MARKER_CROSS_CATEGORIES = ["COLOUR_MARKER_CROSS"]

_ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _apply_marker_cross(inp: list[list[int]]) -> list[list[int]]:
    """Apply the colour-marker cross rule."""
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]

    for r in range(H):
        for c in range(W):
            if inp[r][c] == 1:
                # Paint orthogonal neighbours colour 7
                for dr, dc in _ORTHO:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = 7
            elif inp[r][c] == 2:
                # Paint diagonal neighbours colour 4
                for dr, dc in _DIAG:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = 4

    return out


def detect_colour_marker_cross(task: dict) -> bool:
    """Return True if every training pair matches the colour-marker cross rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        # Must have at least one 1 or 2 cell
        has_marker = any(inp[r][c] in (1, 2) for r in range(H) for c in range(W))
        if not has_marker:
            return False

        expected = _apply_marker_cross(inp)
        if expected != out:
            return False

    return True


def solve_colour_marker_cross(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply colour-marker cross rule: 1→+7, 2→x4."""
    H, W = len(input_grid), len(input_grid[0])
    has_marker = any(input_grid[r][c] in (1, 2) for r in range(H) for c in range(W))
    if not has_marker:
        return None
    return _apply_marker_cross(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_colour_marker_cross(task: dict) -> list[str]:
    """Return ['COLOUR_MARKER_CROSS'] if the task matches the marker-cross rule."""
    return COLOUR_MARKER_CROSS_CATEGORIES if detect_colour_marker_cross(task) else []
