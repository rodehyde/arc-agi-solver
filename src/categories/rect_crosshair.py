"""
RECT_CROSSHAIR (41e4d17e training)

Input: a background of 8s containing hollow rectangles (1-colored borders with
8-interior). Each rectangle has a unique center interior cell.

Rule: for each hollow rectangle, find its interior center (center row =
(r1+r2)//2, center col = (c1+c2)//2). Draw a full-grid crosshair of 6s
through that center, overwriting only 8 cells (1 walls are unchanged).

Detection: at least one hollow 1-rectangle; prediction matches all training pairs.
"""


def _find_rects(inp):
    H, W = len(inp), len(inp[0])
    visited = set()
    rects = []
    for r1 in range(H):
        for c1 in range(W):
            if inp[r1][c1] != 1:
                continue
            c2 = c1
            while c2 + 1 < W and inp[r1][c2 + 1] == 1:
                c2 += 1
            if c2 == c1:
                continue
            r2 = r1
            while r2 + 1 < H and inp[r2 + 1][c1] == 1:
                r2 += 1
            if r2 == r1:
                continue
            valid = all(inp[r1][c] == 1 and inp[r2][c] == 1 for c in range(c1, c2 + 1))
            if valid:
                valid = all(inp[r][c1] == 1 and inp[r][c2] == 1 for r in range(r1, r2 + 1))
            if valid:
                valid = all(inp[r][c] == 8
                            for r in range(r1 + 1, r2)
                            for c in range(c1 + 1, c2))
            if valid:
                key = (r1, c1, r2, c2)
                if key not in visited:
                    visited.add(key)
                    rects.append(key)
    return rects


def _predict(inp):
    H, W = len(inp), len(inp[0])
    rects = _find_rects(inp)
    if not rects:
        return None
    out = [list(row) for row in inp]
    for r1, c1, r2, c2 in rects:
        cr = (r1 + r2) // 2
        cc = (c1 + c2) // 2
        for c in range(W):
            if out[cr][c] == 8:
                out[cr][c] = 6
        for r in range(H):
            if out[r][cc] == 8:
                out[r][cc] = 6
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
