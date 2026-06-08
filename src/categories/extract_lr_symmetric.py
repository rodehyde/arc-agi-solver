"""
EXTRACT_LR_SYMMETRIC (d56f2372 evaluation)

Input: a large grid containing several colored shapes (one color per shape).

Rule: find the unique shape whose bounding box is left-right symmetric
(every row reads the same forwards and backwards); output that bounding box.

Detection: exactly one color's bounding-box sub-grid is LR-symmetric;
prediction matches all training pairs.
"""

from collections import defaultdict


def _get_shapes(grid):
    H, W = len(grid), len(grid[0])
    colors = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                colors[grid[r][c]].append((r, c))
    shapes = {}
    for color, cells in colors.items():
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
        sub = [
            [grid[r][c] if grid[r][c] == color else 0
             for c in range(c1, c2 + 1)]
            for r in range(r1, r2 + 1)
        ]
        shapes[color] = sub
    return shapes


def _is_lr_symmetric(sub):
    return all(row == row[::-1] for row in sub)


def _predict(inp):
    shapes = _get_shapes(inp)
    lr_syms = [(color, sub) for color, sub in shapes.items()
               if _is_lr_symmetric(sub)]
    if len(lr_syms) != 1:
        return None
    return lr_syms[0][1]


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        predicted = _predict(inp)
        if predicted is None or predicted != out:
            return False
    return True


def solve(inp):
    result = _predict(inp)
    return result if result is not None else [list(row) for row in inp]


def categorise(task):
    return detect(task)
