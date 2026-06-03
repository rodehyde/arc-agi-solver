"""
hole_fill_2x2.py — Detection and solving of 2x2 hole-fill tasks.

A task matches HOLE_FILL_2x2 if:
  1. Every training input is a dense grid mostly filled with colour 5, with scattered 0s.
  2. The output fills with colour 2 every cell that belongs to at least one 2x2 all-zero
     sub-region.

Example task: a8d7556c
"""

HOLE_FILL_2X2_CATEGORIES = ["HOLE_FILL_2X2"]


def _cells_in_any_2x2_zero(inp: list[list[int]]) -> set[tuple[int, int]]:
    """
    Return the set of (r, c) positions that belong to at least one 2x2 all-zero block.
    """
    H, W = len(inp), len(inp[0])
    filled: set[tuple[int, int]] = set()

    for r in range(H - 1):
        for c in range(W - 1):
            if (
                inp[r][c] == 0
                and inp[r][c + 1] == 0
                and inp[r + 1][c] == 0
                and inp[r + 1][c + 1] == 0
            ):
                filled.add((r, c))
                filled.add((r, c + 1))
                filled.add((r + 1, c))
                filled.add((r + 1, c + 1))

    return filled


def _apply_hole_fill(inp: list[list[int]]) -> list[list[int]]:
    """Fill cells belonging to any 2x2 all-zero block with colour 2."""
    out = [row[:] for row in inp]
    for r, c in _cells_in_any_2x2_zero(inp):
        out[r][c] = 2
    return out


def detect_hole_fill_2x2(task: dict) -> bool:
    """Return True if every training pair matches the 2x2 hole-fill rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        # Verify changed cells: 0 in input → 2 in output
        changed_cells = _cells_in_any_2x2_zero(inp)

        for r in range(H):
            for c in range(W):
                if inp[r][c] == 0 and (r, c) in changed_cells:
                    if out[r][c] != 2:
                        return False
                else:
                    if out[r][c] != inp[r][c]:
                        return False

    return True


def solve_hole_fill_2x2(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Fill every cell in a 2x2 all-zero block with colour 2."""
    return _apply_hole_fill(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_hole_fill_2x2(task: dict) -> list[str]:
    """Return ['HOLE_FILL_2X2'] if the task matches the 2x2 hole-fill rule."""
    return HOLE_FILL_2X2_CATEGORIES if detect_hole_fill_2x2(task) else []
