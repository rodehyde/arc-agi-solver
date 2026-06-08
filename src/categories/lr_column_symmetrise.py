"""
LR_COLUMN_SYMMETRISE (c35c1b4c training)

The input contains a large connected blob of a key color, embedded in noise
(other non-background colors scattered around). The blob is almost left-right
symmetric but has missing cells on one or both sides.

Rule: for each column c, the output key-color rows = union of key-color rows
in column c and its mirror column (W-1-c). Noise cells that fall in those
rows are overwritten with the key color.

Detection: every training pair has exactly one color that appears in the output
but not the input (the "added" color), and for every pair the column-union
prediction matches the actual output exactly.
"""


def _key_color(inp, out):
    """Return the color that gets added in the output (0 = not found)."""
    H, W = len(inp), len(inp[0])
    for r in range(H):
        for c in range(W):
            if out[r][c] != inp[r][c]:
                return out[r][c]
    return 0


def _col_rows(inp, key, W, H):
    return {c: {r for r in range(H) if inp[r][c] == key} for c in range(W)}


def _predicted_adds(inp, key):
    H, W = len(inp), len(inp[0])
    cr = _col_rows(inp, key, W, H)
    adds = set()
    for c in range(W):
        mirror = W - 1 - c
        union = cr[c] | cr[mirror]
        for r in union:
            if inp[r][c] != key:
                adds.add((r, c))
    return adds


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        H, W = len(inp), len(inp[0])
        if W != 10:
            return False
        key = _key_color(inp, out)
        if key == 0:
            return False
        # All changes must be inp[r][c] -> key
        for r in range(H):
            for c in range(W):
                if inp[r][c] != out[r][c] and out[r][c] != key:
                    return False
        # Column-union must predict changes exactly
        actual = {(r, c) for r in range(H) for c in range(W)
                  if inp[r][c] != out[r][c]}
        if _predicted_adds(inp, key) != actual:
            return False
    return True


def solve(inp):
    H, W = len(inp), len(inp[0])
    # The key color is the dominant blob color — the most frequent non-zero color.
    from collections import Counter
    counts = Counter(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
    if not counts:
        return [list(row) for row in inp]
    key = counts.most_common(1)[0][0]
    adds = _predicted_adds(inp, key)
    out = [list(row) for row in inp]
    for r, c in adds:
        out[r][c] = key
    return out


def categorise(task):
    return detect(task)
