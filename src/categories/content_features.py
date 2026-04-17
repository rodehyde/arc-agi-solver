"""
content_features.py — Categorise ARC tasks based on grid content and colour features.

Categories:
    BACKGROUND_PRESERVED      — dominant colour is the same in every input and output
    DISTINCT_COLOUR_COUNT_SAME — number of unique colours is the same in every pair
    SINGLE_COLOUR_OUTPUT       — each output uses at most one non-background colour
    OBJECT_COUNT_PRESERVED     — number of connected non-background blobs matches in every pair
    OUTPUT_HAS_SYMMETRY        — every output is horizontally or vertically symmetric
    COLOUR_PERMUTATION         — output is the input with a consistent colour remapping
"""

from collections import Counter


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _modal_colour(grid: list[list[int]]) -> int:
    """Return the most frequent colour value in a grid (used as background)."""
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _distinct_colours(grid: list[list[int]]) -> int:
    """Return the number of distinct colour values present in a grid."""
    return len({c for row in grid for c in row})


def _connected_components(grid: list[list[int]]) -> int:
    """Count non-background connected blobs (4-connectivity)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    bg = _modal_colour(grid)
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                count += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr][cc] or grid[cr][cc] == bg:
                        continue
                    visited[cr][cc] = True
                    stack += [(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)]
    return count


def _is_symmetric_h(grid: list[list[int]]) -> bool:
    """True if the grid is symmetric about its horizontal mid-axis (top/bottom mirror)."""
    return grid == grid[::-1]


def _is_symmetric_v(grid: list[list[int]]) -> bool:
    """True if the grid is symmetric about its vertical mid-axis (left/right mirror)."""
    return all(row == row[::-1] for row in grid)


# ---------------------------------------------------------------------------
# Category detectors
# ---------------------------------------------------------------------------

CONTENT_CATEGORIES = [
    "BACKGROUND_PRESERVED",
    "DISTINCT_COLOUR_COUNT_SAME",
    "SINGLE_COLOUR_OUTPUT",
    "OBJECT_COUNT_PRESERVED",
    "OUTPUT_HAS_SYMMETRY",
    "COLOUR_PERMUTATION",
]


def categorise_content(task: dict) -> list[str]:
    """
    Return content-based category labels that apply to this task,
    based on its training examples only.
    """
    pairs = task["train"]
    categories = []

    # --- BACKGROUND_PRESERVED: modal colour unchanged across every pair ---
    if all(_modal_colour(p["input"]) == _modal_colour(p["output"]) for p in pairs):
        categories.append("BACKGROUND_PRESERVED")

    # --- DISTINCT_COLOUR_COUNT_SAME: number of unique colours matches ---
    if all(_distinct_colours(p["input"]) == _distinct_colours(p["output"]) for p in pairs):
        categories.append("DISTINCT_COLOUR_COUNT_SAME")

    # --- SINGLE_COLOUR_OUTPUT: output uses at most one non-background colour ---
    if all(_distinct_colours(p["output"]) <= 2 for p in pairs):
        categories.append("SINGLE_COLOUR_OUTPUT")

    # --- OBJECT_COUNT_PRESERVED: connected blob count matches in every pair ---
    if all(_connected_components(p["input"]) == _connected_components(p["output"])
           for p in pairs):
        categories.append("OBJECT_COUNT_PRESERVED")

    # --- OUTPUT_HAS_SYMMETRY: every output is H- or V-symmetric ---
    if all(_is_symmetric_h(p["output"]) or _is_symmetric_v(p["output"]) for p in pairs):
        categories.append("OUTPUT_HAS_SYMMETRY")

    # --- COLOUR_PERMUTATION: output is input with a consistent colour remapping ---
    same_dims = all(
        len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])
        for p in pairs
    )
    if same_dims:
        is_perm = True
        for p in pairs:
            mapping: dict[int, int] = {}
            for row_in, row_out in zip(p["input"], p["output"]):
                for ci, co in zip(row_in, row_out):
                    if ci in mapping and mapping[ci] != co:
                        is_perm = False
                        break
                    mapping[ci] = co
                if not is_perm:
                    break
            if not is_perm:
                break
        if is_perm:
            categories.append("COLOUR_PERMUTATION")

    return categories
