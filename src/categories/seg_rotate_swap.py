"""
SEG_ROTATE_SWAP (2601afb7 training)

Input: background of 7s with several vertical segments of different colors at
distinct columns. Each segment is bottom-anchored (fills rows H-length to H-1
with a single color).

Rule: let the segments be indexed left to right as 0..n-1.
  - Output segment k gets COLOR from input segment k-1 (rotate right by 1).
  - Output segment k gets LENGTH from input segment k+1 (rotate left by 1).
Segments stay at the same columns; each is still bottom-anchored.

Detection: all segment cells are uniform per column; no partial fills; rotate
prediction matches every training pair.
"""


def _get_segments(inp):
    H, W = len(inp), len(inp[0])
    bg = 7
    segs = []
    for c in range(W):
        col_vals = [inp[r][c] for r in range(H) if inp[r][c] != bg]
        if col_vals and all(v == col_vals[0] for v in col_vals):
            segs.append((c, col_vals[0], len(col_vals)))
    return segs


def _predict(inp):
    H, W = len(inp), len(inp[0])
    bg = 7
    segs = _get_segments(inp)
    if not segs:
        return None
    n = len(segs)
    cols = [s[0] for s in segs]
    colors = [s[1] for s in segs]
    lengths = [s[2] for s in segs]

    new_colors = [colors[(i - 1) % n] for i in range(n)]
    new_lengths = [lengths[(i + 1) % n] for i in range(n)]

    out = [[bg] * W for _ in range(H)]
    for i in range(n):
        c = cols[i]
        color = new_colors[i]
        length = new_lengths[i]
        for r in range(H - length, H):
            out[r][c] = color
    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        predicted = _predict(inp)
        if predicted is None or predicted != out:
            return False
    return True


def solve(inp):
    result = _predict(inp)
    return result if result is not None else [list(row) for row in inp]


def categorise(task):
    return detect(task)
