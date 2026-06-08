"""
T_POINTER_STACK (4c177718 evaluation)

Input: 15×15 grid split by a full row of 5s into a legend section (above) and
query section (below, 9 rows). Legend contains 3 shapes: color-1 (left),
color-2 T-shape (center), and a third color (right). The query section
contains a single color-1 shape matching the legend's left shape by geometry.

Rule:
  - The T-shape (color 2) acts as a pointer. If its solid bar is at the bottom
    row, shape3 goes immediately ABOVE the query; if bar at top, shape3 goes
    immediately BELOW.
  - Output = the 9-row query section with shape3 placed at the query's columns,
    adjacent to the query in the indicated direction.

Detection: separator row of 5s exists; legend has colors 1, 2, and one other;
prediction matches all training pairs.
"""

from collections import defaultdict


def _get_separator_row(grid):
    H, W = len(grid), len(grid[0])
    for r in range(H):
        if all(grid[r][c] == 5 for c in range(W)):
            return r
    return None


def _section_shapes(grid, row_start, row_end):
    cells = defaultdict(list)
    for r in range(row_start, row_end):
        for c in range(len(grid[r])):
            v = grid[r][c]
            if v not in (0, 5):
                cells[v].append((r - row_start, c))
    return dict(cells)


def _sub_grid(cells, color):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
    h, w = r2 - r1 + 1, c2 - c1 + 1
    g = [[0] * w for _ in range(h)]
    for r, c in cells:
        g[r - r1][c - c1] = color
    return g, r1, r2, c1, c2


def _predict(inp):
    H, W = len(inp), len(inp[0])
    sep = _get_separator_row(inp)
    if sep is None:
        return None
    legend = _section_shapes(inp, 0, sep)
    lower = _section_shapes(inp, sep + 1, H)
    if 2 not in legend or 1 not in legend or 1 not in lower:
        return None

    # Determine T direction from color 2
    c2_cells = legend[2]
    rows_of_2 = defaultdict(list)
    for r, c in c2_cells:
        rows_of_2[r].append(c)
    bar_row = max(rows_of_2, key=lambda r: len(rows_of_2[r]))
    max_r2 = max(r for r, c in c2_cells)
    shape3_above = (bar_row == max_r2)  # bar at bottom → shape3 above

    shape3_colors = [c for c in legend if c not in (1, 2)]
    if len(shape3_colors) != 1:
        return None
    s3c = shape3_colors[0]
    s3_grid, _, _, _, _ = _sub_grid(legend[s3c], s3c)

    q_cells = lower[1]
    q_grid, r1q, r2q, c1q, c2q = _sub_grid(q_cells, 1)

    out_H = H - sep - 1
    out = [[0] * W for _ in range(out_H)]
    for r in range(len(q_grid)):
        for c in range(len(q_grid[0])):
            out[r1q + r][c1q + c] = q_grid[r][c]

    s3_r = (r1q - len(s3_grid)) if shape3_above else (r2q + 1)
    for r in range(len(s3_grid)):
        for c in range(len(s3_grid[0])):
            out[s3_r + r][c1q + c] = s3_grid[r][c]
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
