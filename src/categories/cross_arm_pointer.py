"""
CROSS_ARM_POINTER (689c358e training)

Input: a bordered grid (color 6) with interior background (color 7) containing
several plus-shaped crosses of different colors. Each cross has exactly one
asymmetric axis — one arm is length 2 and the opposite arm is length 1.

Rule: for each cross, find the asymmetric axis.
  - Place the cross's color on the border at the SHORT arm's direction.
  - Place color 0 on the border at the LONG arm's direction.

Detection: exactly one non-6/non-7 color per distinct cross; each cross has a
unique center cell (in both a horizontal and vertical run); exactly one pair of
opposite arms differs in length; the predicted border marks match the actual
output for all training pairs.
"""

from collections import defaultdict


def _find_crosses(inp):
    """Return list of (color, center_r, center_c, up, down, left, right) per cross."""
    H, W = len(inp), len(inp[0])
    cells_by_color = defaultdict(list)
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 7 and v != 6:
                cells_by_color[v].append((r, c))

    crosses = []
    for color, cells in cells_by_color.items():
        row_counts = defaultdict(list)
        col_counts = defaultdict(list)
        for r, c in cells:
            row_counts[r].append(c)
            col_counts[c].append(r)

        center = None
        for r, c in cells:
            if len(row_counts[r]) > 1 and len(col_counts[c]) > 1:
                center = (r, c)
                break
        if center is None:
            continue

        cr, cc = center
        up    = sum(1 for r, c in cells if c == cc and r < cr)
        down  = sum(1 for r, c in cells if c == cc and r > cr)
        left  = sum(1 for r, c in cells if r == cr and c < cc)
        right = sum(1 for r, c in cells if r == cr and c > cc)
        crosses.append((color, cr, cc, up, down, left, right))
    return crosses


def _predict(inp):
    H, W = len(inp), len(inp[0])
    crosses = _find_crosses(inp)
    if not crosses:
        return None

    out = [list(row) for row in inp]
    for color, cr, cc, up, down, left, right in crosses:
        if up != down:
            if up < down:
                out[0][cc] = color
                out[H - 1][cc] = 0
            else:
                out[H - 1][cc] = color
                out[0][cc] = 0
        elif left != right:
            if left < right:
                out[cr][0] = color
                out[cr][W - 1] = 0
            else:
                out[cr][W - 1] = color
                out[cr][0] = 0
    return out


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
