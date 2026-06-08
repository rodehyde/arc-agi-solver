"""
MATRIX_BLOCK_EXPAND (33067df9 training)

Input: a small grid whose non-zero cells form an n_rows×n_cols matrix
(at odd rows and odd columns, separated by 0-rows and 0-columns).

Rule: expand the matrix into a fixed 26×26 output.
  - Block dimensions: block_h = 24//n_rows − 2, block_w = 24//n_cols − 2.
  - Blocks are placed at row_start[i] = 2 + i*(block_h+2), col_start[j] = 2 + j*(block_w+2).
  - Within each row, consecutive same-color cells merge horizontally (the gap
    between them disappears, making a wider block).
  - Between adjacent matrix rows: if a run (same color, same column span)
    appears in both rows i and i+1, the 2-row gap between them is filled too
    (vertical merge).

Detection: output is always 26×26; prediction matches every training pair.
"""


def _get_runs(row):
    """List of (color, start_j, end_j) for non-zero runs."""
    runs = []
    j, n = 0, len(row)
    while j < n:
        c = row[j]
        if c == 0:
            j += 1
            continue
        k = j
        while k + 1 < n and row[k + 1] == c:
            k += 1
        runs.append((c, j, k))
        j = k + 1
    return runs


def _predict(inp):
    H, W = len(inp), len(inp[0])
    matrix = [[inp[r][c] for c in range(1, W, 2)] for r in range(1, H, 2)]
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if n_rows == 0 or n_cols == 0:
        return None

    out = [[0] * 26 for _ in range(26)]
    block_h = 24 // n_rows - 2
    block_w = 24 // n_cols - 2
    row_starts = [2 + i * (block_h + 2) for i in range(n_rows)]
    col_starts = [2 + j * (block_w + 2) for j in range(n_cols)]

    # Place horizontal blocks
    for i, row in enumerate(matrix):
        rs = row_starts[i]
        re = rs + block_h - 1
        for color, js, je in _get_runs(row):
            cs = col_starts[js]
            ce = col_starts[je] + block_w - 1
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    out[r][c] = color

    # Vertical merge: identical run in adjacent rows fills the gap
    for i in range(n_rows - 1):
        gap_rs = row_starts[i] + block_h
        gap_re = row_starts[i + 1] - 1
        set_i1 = set(_get_runs(matrix[i + 1]))
        for run in _get_runs(matrix[i]):
            if run in set_i1:
                color, js, je = run
                cs = col_starts[js]
                ce = col_starts[je] + block_w - 1
                for r in range(gap_rs, gap_re + 1):
                    for c in range(cs, ce + 1):
                        out[r][c] = color

    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if len(out) != 26 or len(out[0]) != 26:
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
