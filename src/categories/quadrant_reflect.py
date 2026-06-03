"""
quadrant_reflect.py — Detection and solving of quadrant-reflect tasks.

A task matches QUADRANT_REFLECT if:
  1. The input has a separator cross: one full-width row and one full-height column
     of a single separator colour, dividing the grid into 4 quadrants.
  2. Exactly one quadrant contains a non-zero, non-separator shape; the other three
     are empty (all 0).
  3. Output:
     - Size = 2 * quadrant_height × 2 * quadrant_width (separator lines removed).
     - The shape is recoloured to the separator colour.
     - Placed in all four quadrants:
         top-left    = original orientation (recoloured)
         top-right   = flip_h (mirror left-right)
         bottom-left = flip_v (mirror top-bottom)
         bottom-right= rot180

Example task: 47c1f68c
"""

QUADRANT_REFLECT_CATEGORIES = ["QUADRANT_REFLECT"]


def _find_separator(inp: list[list[int]]) -> tuple[int, int, int] | None:
    """
    Find the separator row, separator column, and separator colour.
    Returns (sep_row, sep_col, sep_colour) or None if not found.
    """
    H, W = len(inp), len(inp[0])

    # Find full-width rows that are all the same non-zero colour
    sep_rows = []
    for r in range(H):
        first = inp[r][0]
        if first != 0 and all(inp[r][c] == first for c in range(W)):
            sep_rows.append((r, first))

    # Find full-height columns that are all the same non-zero colour
    sep_cols = []
    for c in range(W):
        first = inp[0][c]
        if first != 0 and all(inp[r][c] == first for r in range(H)):
            sep_cols.append((c, first))

    if len(sep_rows) != 1 or len(sep_cols) != 1:
        return None

    sep_row, colour_r = sep_rows[0]
    sep_col, colour_c = sep_cols[0]

    if colour_r != colour_c:
        return None

    return sep_row, sep_col, colour_r


def _extract_quadrant(
    inp: list[list[int]],
    r_lo: int,
    r_hi: int,
    c_lo: int,
    c_hi: int,
) -> list[list[int]]:
    """Extract a rectangular sub-region of the grid."""
    return [inp[r][c_lo : c_hi + 1] for r in range(r_lo, r_hi + 1)]


def _is_empty(quad: list[list[int]]) -> bool:
    """Return True if all cells are 0."""
    return all(v == 0 for row in quad for v in row)


def _recolour(quad: list[list[int]], colour: int) -> list[list[int]]:
    """Replace all non-zero values with colour."""
    return [[colour if v != 0 else 0 for v in row] for row in quad]


def _flip_h(quad: list[list[int]]) -> list[list[int]]:
    """Mirror left-right."""
    return [row[::-1] for row in quad]


def _flip_v(quad: list[list[int]]) -> list[list[int]]:
    """Mirror top-bottom."""
    return quad[::-1]


def _rot180(quad: list[list[int]]) -> list[list[int]]:
    """Rotate 180 degrees."""
    return [row[::-1] for row in quad[::-1]]


def _build_quadrant_reflect(inp: list[list[int]]) -> list[list[int]] | None:
    """Build the reflected output. Returns None if the pattern doesn't apply."""
    sep = _find_separator(inp)
    if sep is None:
        return None

    sep_row, sep_col, sep_colour = sep
    H, W = len(inp), len(inp[0])

    # Define the four quadrant regions
    quads = {
        "TL": _extract_quadrant(inp, 0, sep_row - 1, 0, sep_col - 1),
        "TR": _extract_quadrant(inp, 0, sep_row - 1, sep_col + 1, W - 1),
        "BL": _extract_quadrant(inp, sep_row + 1, H - 1, 0, sep_col - 1),
        "BR": _extract_quadrant(inp, sep_row + 1, H - 1, sep_col + 1, W - 1),
    }

    # All quadrants must have the same dimensions
    heights = {len(q) for q in quads.values()}
    widths = {len(q[0]) if q else 0 for q in quads.values()}
    if len(heights) != 1 or len(widths) != 1:
        return None

    qH = heights.pop()
    qW = widths.pop()

    if qH == 0 or qW == 0:
        return None

    # Exactly one quadrant must be non-empty
    non_empty = [(name, q) for name, q in quads.items() if not _is_empty(q)]
    if len(non_empty) != 1:
        return None

    name, shape = non_empty[0]

    # Recolour the shape to sep_colour
    shape_rc = _recolour(shape, sep_colour)

    # Produce the four versions
    tl = shape_rc
    tr = _flip_h(shape_rc)
    bl = _flip_v(shape_rc)
    br = _rot180(shape_rc)

    # But: the top-left placement depends on which quadrant the shape was in.
    # The shape in TL position goes into the TL output quadrant as-is (original).
    # If the shape was in TR, the "original orientation" in TL means the shape
    # as it appears in the top-left quadrant — which would be the flip_h of TR.
    # The rule says: top-left = original orientation.
    # "Original orientation" = the shape as extracted from the populated quadrant,
    # placed in TL.
    #
    # Map from populated quadrant to what appears in TL:
    if name == "TL":
        tl_shape = shape_rc
    elif name == "TR":
        tl_shape = _flip_h(shape_rc)
    elif name == "BL":
        tl_shape = _flip_v(shape_rc)
    elif name == "BR":
        tl_shape = _rot180(shape_rc)
    else:
        return None

    tl = tl_shape
    tr = _flip_h(tl_shape)
    bl = _flip_v(tl_shape)
    br = _rot180(tl_shape)

    # Assemble the output: 2*qH rows x 2*qW cols
    out = []
    for r in range(qH):
        out.append(tl[r] + tr[r])
    for r in range(qH):
        out.append(bl[r] + br[r])

    return out


def detect_quadrant_reflect(task: dict) -> bool:
    """Return True if every training pair matches the quadrant-reflect rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]

        expected = _build_quadrant_reflect(inp)
        if expected is None:
            return False
        if len(out) != len(expected) or (out and len(out[0]) != len(expected[0])):
            return False
        if expected != out:
            return False

    return True


def solve_quadrant_reflect(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the quadrant-reflect rule."""
    return _build_quadrant_reflect(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_quadrant_reflect(task: dict) -> list[str]:
    """Return ['QUADRANT_REFLECT'] if the task matches the rule."""
    return QUADRANT_REFLECT_CATEGORIES if detect_quadrant_reflect(task) else []
