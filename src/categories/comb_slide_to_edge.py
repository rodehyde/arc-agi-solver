"""
COMB_SLIDE_TO_EDGE (9f41bd9c training)

Input: a grid divided by a horizontal separator (all 6s). Above the separator
is a 1-region containing a "comb" shape (5s): a solid rectangular top
connected to alternating vertical teeth. The comb is flush against one
horizontal edge of the 1-region.

Rule:
  1. The comb slides to the OPPOSITE horizontal edge (mirror across the grid).
  2. Its solid rows stay at the same height; the tooth rows trail diagonally
     at +1 (if comb moved left → teeth shift right each row) or −1 (right → left).
  3. The separator row is updated: 6 stays where the trailing-edge teeth have
     NOT swept, and the rest becomes 9.

Detection: one color of 5 forms the comb, one color of 6 forms the separator;
the prediction matches the output for all training pairs.
"""


def _predict(inp):
    H, W = len(inp), len(inp[0])

    comb_cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 5]
    if not comb_cells:
        return None

    min_r = min(r for r, c in comb_cells)
    max_r = max(r for r, c in comb_cells)
    min_c = min(c for r, c in comb_cells)
    max_c = max(c for r, c in comb_cells)
    comb_width = max_c - min_c + 1

    sep_row = None
    for r in range(H):
        if all(inp[r][c] == 6 for c in range(W)):
            sep_row = r
            break
    if sep_row is None:
        return None

    if min_c == 0:
        new_start_c = W - comb_width
        direction = -1
    else:
        new_start_c = 0
        direction = +1

    solid_drs = []
    tooth_rows = []
    for dr in range(max_r - min_r + 1):
        r = min_r + dr
        rel_cols = [c - min_c for c in range(W) if inp[r][c] == 5]
        if len(rel_cols) == comb_width:
            solid_drs.append(dr)
        else:
            tooth_rows.append((dr, rel_cols))

    out = [list(row) for row in inp]
    for r, c in comb_cells:
        out[r][c] = 1

    for dr in solid_drs:
        r = min_r + dr
        for c in range(new_start_c, new_start_c + comb_width):
            out[r][c] = 5

    if tooth_rows:
        first_tooth_row = min_r + tooth_rows[0][0]
        for dr, rel_cols in tooth_rows:
            r = min_r + dr
            shift = (r - first_tooth_row) * direction
            for col_off in rel_cols:
                c = new_start_c + col_off + shift
                if 0 <= c < W:
                    out[r][c] = 5

        last_dr, last_rel_cols = tooth_rows[-1]
        last_r = min_r + last_dr
        last_shift = (last_r - first_tooth_row) * direction
        last_tooth_abs = [new_start_c + col_off + last_shift
                          for col_off in last_rel_cols
                          if 0 <= new_start_c + col_off + last_shift < W]

        if last_tooth_abs:
            if direction == +1:
                leftmost = min(last_tooth_abs)
                for c in range(W):
                    out[sep_row][c] = 6 if c <= leftmost else 9
            else:
                rightmost = max(last_tooth_abs)
                for c in range(W):
                    out[sep_row][c] = 9 if c < rightmost else 6

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
