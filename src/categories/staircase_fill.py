"""
STAIRCASE_FILL (1e97544e evaluation)

Input: H×W grid (H == W) where cells follow a staircase pattern but
some rectangular regions are zeroed out. The staircase rule for the
complete grid is:
    output[r][c] = toprow[c]     if c >= r
    output[r][c] = toprow[c+1]   if c < r
where toprow is a fixed H-length sequence (the top row of the completed
staircase, typically a repeating color sequence).

Rule: reconstruct toprow from the non-zero input cells, then fill every
cell using the staircase formula.

Detection: toprow reconstructed from non-zero cells yields a consistent
staircase that matches output on all training pairs.
"""


def _reconstruct_toprow(inp):
    H, W = len(inp), len(inp[0])
    toprow = [0] * W
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0:
                if c >= r:
                    toprow[c] = v
                elif c + 1 < W:
                    toprow[c + 1] = v
    return toprow


def _predict(inp):
    H, W = len(inp), len(inp[0])
    toprow = _reconstruct_toprow(inp)
    if 0 in toprow:
        return None
    out = []
    for r in range(H):
        row = []
        for c in range(W):
            if c >= r:
                row.append(toprow[c])
            else:
                row.append(toprow[c + 1] if c + 1 < W else toprow[c])
        out.append(row)
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
