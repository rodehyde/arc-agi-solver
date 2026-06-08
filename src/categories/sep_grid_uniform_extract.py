"""
SEP_GRID_UNIFORM_EXTRACT (458e3a53 training)

Input: a large grid with regular separator lines (full rows and columns of a
single repeating color) dividing it into a grid of cells. Most cells contain a
tile background (varied values); some cells are uniformly filled with a solid
color.

Rule: find the contiguous rectangular block of uniformly-filled cells; output
their colors as a 2D matrix.

Detection: separator rows/cols exist; some cells are uniform; prediction
matches all training pairs.
"""


def _get_groups(seps, size):
    prev = -1
    groups = []
    for s in sorted(seps):
        if s > prev + 1:
            groups.append((prev + 1, s - 1))
        prev = s
    if prev + 1 < size:
        groups.append((prev + 1, size - 1))
    return groups


def _predict(inp):
    H, W = len(inp), len(inp[0])

    sep_rows = [r for r in range(H) if len(set(inp[r])) == 1]
    sep_cols = [c for c in range(W) if len(set(inp[r][c] for r in range(H))) == 1]

    row_groups = _get_groups(sep_rows, H)
    col_groups = _get_groups(sep_cols, W)

    uniform = {}
    for ri, (r1, r2) in enumerate(row_groups):
        for ci, (c1, c2) in enumerate(col_groups):
            vals = {inp[r][c] for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)}
            if len(vals) == 1:
                uniform[(ri, ci)] = vals.pop()

    if not uniform:
        return None

    min_ri = min(k[0] for k in uniform)
    max_ri = max(k[0] for k in uniform)
    min_ci = min(k[1] for k in uniform)
    max_ci = max(k[1] for k in uniform)

    return [[uniform.get((ri, ci), 0) for ci in range(min_ci, max_ci + 1)]
            for ri in range(min_ri, max_ri + 1)]


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
