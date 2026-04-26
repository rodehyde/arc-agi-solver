"""
transform_features.py — Programmatic detection of input→output transformation types.

These categories are grounded in what changed between input and output,
mirroring the human approach of inspecting the diff first.

Categories
----------
STRUCTURE_UNCHANGED
    The zero/non-zero mask is identical in every input–output pair.
    Covers "no change" (input == output) and "colour-only change"
    (same positions occupied, only colour values differ).

TRANSLATE
    The entire grid content is shifted by one fixed (dr, dc) offset across
    every training pair.  Cells shifted out of bounds become 0.  The shift
    must be consistent (same dr, dc) for all pairs.

CROP
    The output is a literal, unmodified rectangular sub-region of the input
    in every training pair.  Position may vary between pairs.

EXTEND
    Same dimensions; all non-zero input cells are preserved (same colour, same
    position) in the output; at least one new non-zero cell is added.
    This is the broad parent of the two sub-types below.

EXTEND_LINE
    EXTEND holds, and every connected component of added cells forms a straight
    line — horizontal, vertical, or diagonal (slope ±1).

EXTEND_RECT
    EXTEND holds, and every connected component of added cells forms either the
    complete border or the complete fill of an axis-aligned rectangle.
"""

import numpy as np

TRANSFORM_CATEGORIES = [
    "STRUCTURE_UNCHANGED",
    "TRANSLATE",
    "CROP",
    "EXTEND",
    "EXTEND_LINE",
    "EXTEND_RECT",
    "FILL_PATTERN",
]


# ---------------------------------------------------------------------------
# Shared numpy helper
# ---------------------------------------------------------------------------

def _to_np(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int32)


# ---------------------------------------------------------------------------
# STRUCTURE_UNCHANGED helpers
# ---------------------------------------------------------------------------

def _structure_unchanged(inp: np.ndarray, out: np.ndarray) -> bool:
    """True if output has the same shape and same zero/non-zero mask as input."""
    if inp.shape != out.shape:
        return False
    return bool(np.array_equal(inp == 0, out == 0))


# ---------------------------------------------------------------------------
# TRANSLATE helpers
# ---------------------------------------------------------------------------

def _detect_shift(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    """
    Find (dr, dc) such that inp shifted by (dr, dc) matches out exactly.
    Vacated positions become 0.  Returns None if no consistent shift exists.
    """
    if inp.shape != out.shape:
        return None
    H, W = inp.shape
    for dr in range(-(H - 1), H):
        for dc in range(-(W - 1), W):
            if dr == 0 and dc == 0:
                continue
            expected = np.zeros_like(inp)
            r_src = slice(max(0, -dr), min(H, H - dr))
            c_src = slice(max(0, -dc), min(W, W - dc))
            r_dst = slice(max(0,  dr), min(H, H + dr))
            c_dst = slice(max(0,  dc), min(W, W + dc))
            expected[r_dst, c_dst] = inp[r_src, c_src]
            if np.array_equal(expected, out):
                return (dr, dc)
    return None


# ---------------------------------------------------------------------------
# CROP helpers
# ---------------------------------------------------------------------------

def _is_crop(inp: np.ndarray, out: np.ndarray) -> bool:
    """True if output is a literal rectangular sub-region of input."""
    oh, ow = out.shape
    ih, iw = inp.shape
    if oh > ih or ow > iw:
        return False
    for r in range(ih - oh + 1):
        for c in range(iw - ow + 1):
            if np.array_equal(inp[r:r + oh, c:c + ow], out):
                return True
    return False


# ---------------------------------------------------------------------------
# EXTEND helpers
# ---------------------------------------------------------------------------

def _input_preserved(inp: np.ndarray, out: np.ndarray) -> bool:
    """
    True if: same shape, at least one non-zero input cell exists, and every
    non-zero input cell appears unchanged (same colour) at the same position
    in the output.

    Requiring at least one non-zero input cell prevents blank inputs from
    trivially matching EXTEND.
    """
    if inp.shape != out.shape:
        return False
    mask = inp != 0
    if not np.any(mask):
        return False   # blank input → not an extension task
    return bool(np.array_equal(inp[mask], out[mask]))


def _added_cells(inp: np.ndarray, out: np.ndarray) -> list[tuple[int, int]]:
    """(r, c) positions where inp == 0 and out != 0."""
    locs = np.argwhere((inp == 0) & (out != 0))
    return [(int(r), int(c)) for r, c in locs]


def _connected_components(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """4-connected components of a set of (r, c) cells."""
    cell_set = set(cells)
    visited: set[tuple[int, int]] = set()
    components = []
    for seed in cells:
        if seed in visited:
            continue
        comp: list[tuple[int, int]] = []
        stack = [seed]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or (r, c) not in cell_set:
                continue
            visited.add((r, c))
            comp.append((r, c))
            stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        components.append(comp)
    return components


def _is_collinear(cells: list[tuple[int, int]]) -> bool:
    """True if all cells lie on one horizontal, vertical, or diagonal line."""
    if len(cells) <= 1:
        return True
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    if len(set(rows)) == 1:                          # horizontal
        return True
    if len(set(cols)) == 1:                          # vertical
        return True
    if len(set(r - c for r, c in cells)) == 1:      # diagonal (\)
        return True
    if len(set(r + c for r, c in cells)) == 1:      # anti-diagonal (/)
        return True
    return False


def _is_rect_border(cells: list[tuple[int, int]]) -> bool:
    """True if cells form exactly the perimeter of an axis-aligned rectangle."""
    if len(cells) < 4:
        return False
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r0 == r1 or c0 == c1:
        return False  # degenerate (single row or column = a line)
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(cells) == expected


def _is_rect_border_with_input(
    cells: list[tuple[int, int]], inp: np.ndarray
) -> bool:
    """
    True if the added cells, together with any input non-zero cells that lie
    on the bounding-box perimeter, form a complete rectangle border.

    Catches the 'rect-gap' pattern: input corner cells are absent from the
    added set but complete a perfect border when included.
    """
    if not cells:
        return False
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r0 == r1 or c0 == c1:
        return False
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    inp_on_perimeter = {
        (int(r), int(c))
        for r, c in np.argwhere(inp != 0)
        if (int(r), int(c)) in expected
    }
    return set(cells) | inp_on_perimeter == expected


def _is_rect_fill_with_input(
    cells: list[tuple[int, int]], inp: np.ndarray
) -> bool:
    """
    True if the added cells, together with any input non-zero cells that lie
    within the bounding box, form a complete filled rectangle.

    Catches tasks where a filled rectangle is drawn around a small number of
    input anchor/marker cells (e.g. two corner cells of the same colour).

    The input cells inside the box are capped at 6: if there are more, the
    input is a pattern rather than a few markers, and the match is rejected.
    This prevents pattern-fill tasks (where the whole grid is filled and the
    existing pattern cells are numerous) from falsely matching.
    """
    if not cells:
        return False
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    inp_in_box = {
        (int(r), int(c))
        for r, c in np.argwhere(inp != 0)
        if r0 <= int(r) <= r1 and c0 <= int(c) <= c1
    }
    if len(inp_in_box) > 6:      # too many input cells — pattern, not markers
        return False
    return set(cells) | inp_in_box == expected


# ---------------------------------------------------------------------------
# FILL_PATTERN helpers
# ---------------------------------------------------------------------------

def _is_fill_pattern_pair(out: np.ndarray) -> bool:
    """
    True if the output grid is completely dense (all cells non-zero) and can be
    reproduced by tiling a smaller rectangular tile of area < H*W/2.

    Uses ceiling-division tiling so the last row/column tile may be cropped.
    This handles grids whose dimensions are not multiples of the tile period.
    """
    if not np.all(out != 0):
        return False
    H, W = out.shape
    for ht in range(1, H + 1):
        for wt in range(1, W + 1):
            if 2 * ht * wt >= H * W:
                break  # wt only grows; all further wt will also fail
            tile = out[:ht, :wt]
            rh = (H + ht - 1) // ht
            rw = (W + wt - 1) // wt
            tiled = np.tile(tile, (rh, rw))[:H, :W]
            if np.array_equal(tiled, out):
                return True
    return False


def _is_rect_fill(cells: list[tuple[int, int]]) -> bool:
    """True if cells fill an axis-aligned rectangle completely."""
    if not cells:
        return False
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return set(cells) == expected


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------

def categorise_transform(task: dict) -> list[str]:
    """
    Return transformation-based category labels for this task,
    inspecting training pairs only.
    """
    pairs = task["train"]
    np_pairs = [(_to_np(p["input"]), _to_np(p["output"])) for p in pairs]
    categories = []

    # STRUCTURE_UNCHANGED: zero-mask identical across all pairs
    if all(_structure_unchanged(inp, out) for inp, out in np_pairs):
        categories.append("STRUCTURE_UNCHANGED")

    # TRANSLATE: same whole-grid shift applied consistently to every pair
    shifts = [_detect_shift(inp, out) for inp, out in np_pairs]
    if all(s is not None for s in shifts) and len(set(shifts)) == 1:
        categories.append("TRANSLATE")

    # CROP: output is a sub-region of input in every pair
    if all(_is_crop(inp, out) for inp, out in np_pairs):
        categories.append("CROP")

    # EXTEND (parent): same size, input preserved, at least one cell added
    # Build per-pair (inp, preserved, added_cells) only when shapes match.
    extend_data: list[tuple[np.ndarray, bool, list]] = []
    for inp, out in np_pairs:
        preserved = _input_preserved(inp, out)   # False on shape mismatch or blank input
        added = _added_cells(inp, out) if preserved else []
        extend_data.append((inp, preserved, added))

    is_extend = all(preserved and bool(added) for _, preserved, added in extend_data)
    if is_extend:
        categories.append("EXTEND")

        # EXTEND_LINE: every connected component of added cells is a straight line
        def _all_lines(pair_added):
            return all(_is_collinear(comp)
                       for comp in _connected_components(pair_added))

        if all(_all_lines(added) for _, _, added in extend_data):
            categories.append("EXTEND_LINE")

        # EXTEND_RECT: every added component is a rect border or rect fill,
        # with or without input cells completing the corners/interior.
        def _all_rects(pair_inp, pair_added):
            return all(
                _is_rect_border(comp)
                or _is_rect_fill(comp)
                or _is_rect_border_with_input(comp, pair_inp)
                or _is_rect_fill_with_input(comp, pair_inp)
                for comp in _connected_components(pair_added)
            )

        if all(_all_rects(inp, added) for inp, _, added in extend_data):
            categories.append("EXTEND_RECT")

        # FILL_PATTERN: output is fully dense and reproducible by tiling a small tile
        if all(_is_fill_pattern_pair(out) for _, out in np_pairs):
            categories.append("FILL_PATTERN")

    return categories


# ---------------------------------------------------------------------------
# Convenience: detect with detail (useful for inspection / debugging)
# ---------------------------------------------------------------------------

def detect_transform_detail(task: dict) -> dict:
    """
    Return a dict with detected parameters for each category.
    Useful for inspection scripts and notebooks.
    """
    pairs = task["train"]
    np_pairs = [(_to_np(p["input"]), _to_np(p["output"])) for p in pairs]

    unchanged = all(_structure_unchanged(i, o) for i, o in np_pairs)

    shifts = [_detect_shift(i, o) for i, o in np_pairs]
    translate = (shifts[0]
                 if all(s is not None for s in shifts) and len(set(shifts)) == 1
                 else None)

    crop = all(_is_crop(i, o) for i, o in np_pairs)

    extend_checks = [(_input_preserved(i, o), _added_cells(i, o)) for i, o in np_pairs]
    extend = all(p and bool(a) for p, a in extend_checks)

    return {
        "STRUCTURE_UNCHANGED": unchanged,
        "TRANSLATE":           translate,   # (dr, dc) or None
        "CROP":                crop,
        "EXTEND":              extend,
    }
