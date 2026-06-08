"""
APPEND_LR_MIRROR (c9e6f938 training)

Input: any H×W grid.

Rule: output is H × 2W — the original grid concatenated with its left-right
mirror (each row reversed).

Detection: output width = 2 × input width; prediction matches all training pairs.
"""


def _predict(inp):
    H, W = len(inp), len(inp[0])
    out_rows = [row + row[::-1] for row in inp]
    return out_rows


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if len(out) != len(inp) or len(out[0]) != 2 * len(inp[0]):
            return False
        if _predict(inp) != out:
            return False
    return True


def solve(inp):
    return _predict(inp)


def categorise(task):
    return detect(task)
