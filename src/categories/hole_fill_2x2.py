"""
hole_fill_2x2.py — Detection and solving of rectangular-zero-fill tasks.

Rule (two-stage):
  1. Find every cell that belongs to at least one 2×2 all-zero sub-region.
     These are the "candidate" cells.
  2. Group the candidates by 4-connected components.  Within each component,
     find the largest axis-aligned rectangle fully contained in the component
     (using the standard "largest rectangle in histogram" algorithm).
  3. Fill every cell of that rectangle with colour 2.

This naturally handles rectangular zero patches of any size (2×2, 3×2, 4×2,
2×8, etc.) because a larger rectangle always scores higher than any of the
overlapping 2×2 sub-blocks it contains.  When an irregular zero region
contains two competing maximal rectangles, the larger one wins.

Example task: a8d7556c
"""

from collections import deque

HOLE_FILL_2X2_CATEGORIES = ["HOLE_FILL_2X2"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _cells_in_any_2x2_zero(inp: list[list[int]]) -> set[tuple[int, int]]:
    """Return every (r, c) that belongs to at least one 2×2 all-zero block."""
    H, W = len(inp), len(inp[0])
    candidates: set[tuple[int, int]] = set()
    for r in range(H - 1):
        for c in range(W - 1):
            if (inp[r][c] == 0 and inp[r][c + 1] == 0
                    and inp[r + 1][c] == 0 and inp[r + 1][c + 1] == 0):
                candidates.update([(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)])
    return candidates


def _largest_rect_in_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    """
    Find the largest axis-aligned rectangle fully contained in *cells*.

    Uses the standard "largest rectangle in histogram" algorithm row-by-row.
    Returns (min_r, max_r, min_c, max_c) of the best rectangle, or None.
    """
    if not cells:
        return None
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    H = max_r - min_r + 1
    W = max_c - min_c + 1

    mat = [[0] * W for _ in range(H)]
    for r, c in cells:
        mat[r - min_r][c - min_c] = 1

    best_area = 0
    best_rect: tuple[int, int, int, int] | None = None
    heights = [0] * W

    for row in range(H):
        for col in range(W):
            heights[col] = heights[col] + 1 if mat[row][col] else 0

        # Largest rectangle in the current histogram
        stack: list[tuple[int, int]] = []   # (left_boundary, height)
        for i in range(W + 1):
            h = heights[i] if i < W else 0
            left = i
            while stack and stack[-1][1] > h:
                bleft, bh = stack.pop()
                width = i - bleft
                area = bh * width
                if area > best_area:
                    best_area = area
                    best_rect = (
                        min_r + row - bh + 1,
                        min_r + row,
                        min_c + bleft,
                        min_c + bleft + width - 1,
                    )
                left = bleft
            stack.append((left, h))

    return best_rect


def _apply_hole_fill(inp: list[list[int]]) -> list[list[int]]:
    """Apply the rule: fill the largest-rectangle core of each candidate cluster."""
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]

    candidates = _cells_in_any_2x2_zero(inp)
    visited: set[tuple[int, int]] = set()

    for start in candidates:
        if start in visited:
            continue
        # BFS over candidate cells only
        comp: set[tuple[int, int]] = set()
        q: deque[tuple[int, int]] = deque([start])
        while q:
            cell = q.popleft()
            if cell in visited:
                continue
            visited.add(cell)
            comp.add(cell)
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in candidates and nb not in visited:
                    q.append(nb)

        rect = _largest_rect_in_cells(comp)
        if rect is not None:
            r0, r1, c0, c1 = rect
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    out[r][c] = 2

    return out


# ---------------------------------------------------------------------------
# Category / solver interface
# ---------------------------------------------------------------------------

def detect_hole_fill_2x2(task: dict) -> bool:
    """Return True if every training pair matches the rectangular-zero-fill rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        # Normalise to plain lists (task grids may be numpy arrays)
        if hasattr(inp[0], "tolist"):
            inp = [list(row) for row in inp]
        if hasattr(out[0], "tolist"):
            out = [list(row) for row in out]
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return False
        expected = _apply_hole_fill(inp)
        for r in range(H):
            for c in range(W):
                if int(expected[r][c]) != int(out[r][c]):
                    return False
    return True


def solve_hole_fill_2x2(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Fill the largest-rectangle core of each 2×2-zero candidate cluster with colour 2."""
    if not _cells_in_any_2x2_zero(input_grid):
        return None
    return _apply_hole_fill(input_grid)


def categorise_hole_fill_2x2(task: dict) -> list[str]:
    """Return ['HOLE_FILL_2X2'] if the task matches the rule."""
    return HOLE_FILL_2X2_CATEGORIES if detect_hole_fill_2x2(task) else []
