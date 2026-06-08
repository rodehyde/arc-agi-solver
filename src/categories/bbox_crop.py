"""
BBOX_CROP (1cf80156 training)

Input: a large grid of 0s (background) containing a single connected or
disconnected shape in one non-zero color.

Rule: crop to the bounding box of all non-zero cells.

Detection: exactly one non-zero color; output dimensions equal the bounding box
of non-zero cells; prediction matches all training pairs.
"""


def _predict(inp):
    H, W = len(inp), len(inp[0])
    rows = [r for r in range(H) for c in range(W) if inp[r][c] != 0]
    cols = [c for r in range(H) for c in range(W) if inp[r][c] != 0]
    if not rows:
        return None
    r1, r2 = min(rows), max(rows)
    c1, c2 = min(cols), max(cols)
    return [list(inp[r][c1:c2 + 1]) for r in range(r1, r2 + 1)]


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
