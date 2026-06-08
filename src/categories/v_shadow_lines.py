"""
V_SHADOW_LINES (6d58a25d training)

Input: a downward-opening V shape in one color (shape_color) plus scattered
isolated dot cells in a second color (dot_color).

Rule: any dot whose column falls strictly between the V's outermost bottom-arm
columns is a "shadow dot". For each shadow dot column, draw a vertical line of
dot_color starting at (last V-row that contains that column + 1) down to the
grid bottom.

Detection: two non-zero colors per pair; one forms a connected V (bottom row
has exactly 2 cells = the arm tips); the other is isolated dots; and the
column-shadow-line prediction matches the actual output for every pair.
"""

from collections import Counter, defaultdict


def _partition_colors(inp):
    """Return (shape_color, dot_color) by connectivity."""
    H, W = len(inp), len(inp[0])

    def has_neighbor(r, c, v):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] == v:
                return True
        return False

    flat = [v for row in inp for v in row if v != 0]
    if not flat:
        return None, None
    counts = Counter(flat)
    colors = set(counts)
    if len(colors) != 2:
        return None, None
    shape_colors = {v for v in colors
                    if any(has_neighbor(r, c, v)
                           for r in range(H) for c in range(W)
                           if inp[r][c] == v)}
    dot_colors = colors - shape_colors
    if len(shape_colors) != 1 or len(dot_colors) != 1:
        return None, None
    return shape_colors.pop(), dot_colors.pop()


def _build_v(inp, shape_color):
    H, W = len(inp), len(inp[0])
    rows_to_cols = defaultdict(set)
    for r in range(H):
        for c in range(W):
            if inp[r][c] == shape_color:
                rows_to_cols[r].add(c)
    return rows_to_cols


def _shadow_dot_cols(inp, dot_color, left_arm, right_arm, bottom_shape_row):
    H, W = len(inp), len(inp[0])
    shadow = set(range(left_arm + 1, right_arm))
    cols = set()
    for r in range(H):
        for c in range(W):
            if inp[r][c] == dot_color and r > bottom_shape_row and c in shadow:
                cols.add(c)
    return cols


def _predict(inp, shape_color, dot_color):
    H, W = len(inp), len(inp[0])
    rows_to_cols = _build_v(inp, shape_color)
    if not rows_to_cols:
        return None
    bottom_shape_row = max(rows_to_cols)
    bottom_arm_cols = sorted(rows_to_cols[bottom_shape_row])
    if len(bottom_arm_cols) < 2:
        return None
    left_arm, right_arm = bottom_arm_cols[0], bottom_arm_cols[-1]

    sdc = _shadow_dot_cols(inp, dot_color, left_arm, right_arm, bottom_shape_row)
    if not sdc:
        return None

    out = [list(row) for row in inp]
    for col in sdc:
        last_present = max(
            (sr for sr in rows_to_cols if col in rows_to_cols[sr]), default=None
        )
        start_row = (last_present + 1) if last_present is not None else bottom_shape_row + 1
        for r in range(start_row, H):
            out[r][col] = dot_color
    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        shape_color, dot_color = _partition_colors(inp)
        if shape_color is None:
            return False
        predicted = _predict(inp, shape_color, dot_color)
        if predicted is None or predicted != out:
            return False
    return True


def solve(inp):
    shape_color, dot_color = _partition_colors(inp)
    if shape_color is None:
        return [list(row) for row in inp]
    result = _predict(inp, shape_color, dot_color)
    return result if result is not None else [list(row) for row in inp]


def categorise(task):
    return detect(task)
