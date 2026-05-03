"""
flood_fill.py — Programmatic detection of simple enclosed-region flood fill.

A task passes if:
  1. Every training pair has the same output shape as its input.
  2. All cells that change go from background (0) to a single new colour.
  3. That fill colour is the same across every training pair that changes.
  4. Every changed cell is enclosed — not reachable from the grid boundary
     through background-valued cells.

This captures tasks like:
  - "flood-fill enclosed region with colour X"
  - "fill interior of closed border shapes"

It deliberately does NOT cover tasks where multiple fill colours are used
(e.g. different regions get different colours) — those require learning
the assignment rule from context.
"""

from collections import deque

import numpy as np

FLOOD_FILL_CATEGORIES = ["FLOOD_FILL"]

BACKGROUND = 0


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def reachable_from_boundary(grid: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of background cells reachable from the grid boundary.
    Uses BFS through 4-connected background (value == BACKGROUND) cells.
    """
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H - 1 or c == 0 or c == W - 1) and grid[r, c] == BACKGROUND:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == BACKGROUND:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return visited


def enclosed_background(grid: np.ndarray) -> np.ndarray:
    """Return a boolean mask of background cells NOT reachable from the boundary."""
    return (grid == BACKGROUND) & ~reachable_from_boundary(grid)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_flood_fill(task: dict) -> int | None:
    """
    Detect if this task is a simple enclosed-region flood fill.

    Returns the fill colour (int) if detected, or None.

    Detection logic:
      - All pairs must have the same shape in and out.
      - All changed cells must have been BACKGROUND.
      - Changed cells must all be enclosed (not reachable from boundary).
      - The fill colour is a single new value, consistent across all pairs
        that have any changes.
    """
    fill_colour: int | None = None

    for p in task["train"]:
        inp = np.array(p["input"],  dtype=np.int32)
        out = np.array(p["output"], dtype=np.int32)

        if inp.shape != out.shape:
            return None

        changed = inp != out

        # Pairs where output == input are allowed (no enclosed region that pair)
        if not changed.any():
            continue

        # All changed cells must come from background
        if not np.all(inp[changed] == BACKGROUND):
            return None

        # Must be exactly one new colour for this pair
        new_colours = set(int(v) for v in out[changed])
        if len(new_colours) != 1:
            return None

        pair_fill = next(iter(new_colours))

        # Fill colour must be consistent across pairs
        if fill_colour is None:
            fill_colour = pair_fill
        elif fill_colour != pair_fill:
            return None

        # Changed cells must be enclosed in the input — AND the set of
        # enclosed cells must exactly match the set of changed cells.
        # If some enclosed cells are NOT changed, there is a secondary rule
        # that selects which regions to fill; that is out of scope here.
        enc = enclosed_background(inp)
        if not np.array_equal(enc, changed):
            return None

    return fill_colour  # None if every pair was unchanged (degenerate)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_flood_fill(task: dict) -> list[str]:
    """Return ['FLOOD_FILL'] if the task matches simple enclosed-region fill."""
    return FLOOD_FILL_CATEGORIES if detect_flood_fill(task) is not None else []
