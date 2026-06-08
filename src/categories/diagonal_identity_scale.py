"""
DIAGONAL_IDENTITY_SCALE (f0afb749 evaluation)

Input: H×W grid of mostly zeros with a few non-zero colored cells.

Rule: scale to 2H×2W. Each input cell maps to a 2×2 block:
  - Non-zero value V → 2×2 block of V.
  - Zero cell on the same NW→SE diagonal (r−c) as any non-zero cell
    → identity block [[1,0],[0,1]].
  - All other zero cells → 2×2 zero block.

Detection: output is 2× input in both dimensions; prediction matches
all training pairs.
"""


def _predict(inp):
    H, W = len(inp), len(inp[0])
    active_diags = set()
    for r in range(H):
        for c in range(W):
            if inp[r][c] != 0:
                active_diags.add(r - c)
    out = [[0] * (2 * W) for _ in range(2 * H)]
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            br, bc = 2 * r, 2 * c
            if v != 0:
                out[br][bc] = v
                out[br][bc + 1] = v
                out[br + 1][bc] = v
                out[br + 1][bc + 1] = v
            elif (r - c) in active_diags:
                out[br][bc] = 1
                out[br + 1][bc + 1] = 1
    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if len(out) != 2 * len(inp) or len(out[0]) != 2 * len(inp[0]):
            return False
        if _predict(inp) != out:
            return False
    return True


def solve(inp):
    return _predict(inp)


def categorise(task):
    return detect(task)
