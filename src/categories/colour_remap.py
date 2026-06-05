"""
colour_remap.py — Strategy-library approach to colour-remapping tasks.

Each task is tried against a library of candidate strategies in order.
The first strategy whose rule is consistent with ALL training pairs is used
to solve the test input.  This mirrors the detect_transform / detect_logical_op
pattern: enumerate strategies, verify against training pairs, dispatch.

Strategies (tried in order)
----------------------------
static_substitution  Fixed colour map A→B learned from training pairs.
fill_most_common     Every cell → most frequent non-zero colour in the input.
most_common_stays    Most frequent non-zero colour stays; all other non-zero
                     cells → a fixed replacement colour C (learned from training).
concentric_reverse   Concentric rings: reverse the colour sequence.
concentric_rotate    Concentric rings: cyclic rotation — each distinct colour
                     maps to the one belonging to the next outer ring.
corner_indicator     Bottom-left corner cell gives the target colour; the shape
                     (all other non-zero cells) is recoloured to it; corner → 0.
colour5_shape        Cells of colour 5 adopt the other non-zero colour; that
                     other colour → 0.
"""

from collections import Counter

COLOUR_REMAP_CATEGORIES = ["COLOUR_REMAP"]


def _to_list(g):
    if hasattr(g[0], "tolist"):
        return [list(r) for r in g]
    return [list(r) for r in g]


# ── Strategy: static colour substitution ─────────────────────────────────────

def _try_static_substitution(task):
    mapping: dict[int, int] = {}
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        for r in range(H):
            for c in range(W):
                src, dst = inp[r][c], out[r][c]
                if src == dst:
                    continue
                if src == 0 or dst == 0:
                    return None
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    if not mapping:
        return None

    def solve(inp):
        H, W = len(inp), len(inp[0])
        out = [row[:] for row in inp]
        for r in range(H):
            for c in range(W):
                if inp[r][c] in mapping:
                    out[r][c] = mapping[inp[r][c]]
        return out

    return solve


# ── Strategy: fill grid with most common colour ───────────────────────────────

def _try_fill_most_common(task):
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        counts = Counter(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
        if not counts:
            return None
        mc = counts.most_common(1)[0][0]
        if not all(out[r][c] == mc for r in range(H) for c in range(W)):
            return None

    def solve(inp):
        H, W = len(inp), len(inp[0])
        counts = Counter(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
        if not counts:
            return None
        mc = counts.most_common(1)[0][0]
        return [[mc] * W for _ in range(H)]

    return solve


# ── Strategy: most common stays, others → fixed colour ───────────────────────

def _try_most_common_stays(task):
    replace_colour: int | None = None
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        counts = Counter(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
        if not counts:
            return None
        mc = counts.most_common(1)[0][0]
        for r in range(H):
            for c in range(W):
                v, o = inp[r][c], out[r][c]
                if v == mc:
                    if o != mc:
                        return None
                elif v != 0:
                    if replace_colour is None:
                        replace_colour = o
                    elif o != replace_colour:
                        return None
                else:
                    if o != 0:
                        return None
    if replace_colour is None:
        return None
    rc = replace_colour

    def solve(inp):
        H, W = len(inp), len(inp[0])
        counts = Counter(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
        if not counts:
            return None
        mc = counts.most_common(1)[0][0]
        out = [row[:] for row in inp]
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0 and inp[r][c] != mc:
                    out[r][c] = rc
        return out

    return solve


# ── Concentric ring helpers ───────────────────────────────────────────────────

def _ring_sequence(g):
    """Return list of colours per ring depth (outermost first), or None if not uniform."""
    H, W = len(g), len(g[0])
    seq = []
    for d in range(min(H, W) // 2):
        colours = {
            g[r][c]
            for r in range(d, H - d)
            for c in range(d, W - d)
            if min(r - d, c - d, H - 1 - d - r, W - 1 - d - c) == 0
        }
        if len(colours) != 1:
            return None
        seq.append(next(iter(colours)))
    return seq or None


def _apply_ring_colours(g, new_seq):
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    for d, colour in enumerate(new_seq):
        for r in range(d, H - d):
            for c in range(d, W - d):
                if min(r - d, c - d, H - 1 - d - r, W - 1 - d - c) == 0:
                    out[r][c] = colour
    return out


# ── Strategy: concentric rings — reverse sequence ─────────────────────────────

def _try_concentric_reverse(task):
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        if len(out) != len(inp) or len(out[0]) != len(inp[0]):
            return None
        seq_in = _ring_sequence(inp)
        seq_out = _ring_sequence(out)
        if seq_in is None or seq_out is None:
            return None
        if seq_out != list(reversed(seq_in)):
            return None

    def solve(inp):
        seq = _ring_sequence(inp)
        if seq is None:
            return None
        return _apply_ring_colours(inp, list(reversed(seq)))

    return solve


# ── Strategy: concentric rings — cyclic rotation ──────────────────────────────

def _try_concentric_rotate(task):
    """Each distinct ring colour maps to the colour of the next outer ring (cyclic)."""
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        seq = _ring_sequence(inp)
        if seq is None:
            return None
        # Distinct colours in ring order (first occurrence)
        distinct = []
        for c in seq:
            if c not in distinct:
                distinct.append(c)
        if len(distinct) < 2:
            return None
        n = len(distinct)
        mapping = {distinct[i]: distinct[(i - 1) % n] for i in range(n)}
        for r in range(H):
            for c in range(W):
                expected = mapping.get(inp[r][c], inp[r][c])
                if out[r][c] != expected:
                    return None

    def solve(inp):
        H, W = len(inp), len(inp[0])
        seq = _ring_sequence(inp)
        if seq is None:
            return None
        distinct = []
        for c in seq:
            if c not in distinct:
                distinct.append(c)
        n = len(distinct)
        if n < 2:
            return None
        mapping = {distinct[i]: distinct[(i - 1) % n] for i in range(n)}
        out = [row[:] for row in inp]
        for r in range(H):
            for c in range(W):
                if inp[r][c] in mapping:
                    out[r][c] = mapping[inp[r][c]]
        return out

    return solve


# ── Strategy: corner cell indicator ──────────────────────────────────────────

def _try_corner_indicator(task):
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        corner = inp[H - 1][0]
        if corner == 0 or out[H - 1][0] != 0:
            return None
        shape_colours = {
            inp[r][c] for r in range(H) for c in range(W)
            if inp[r][c] != 0 and not (r == H - 1 and c == 0)
        }
        shape_colours.discard(corner)
        if len(shape_colours) != 1:
            return None
        shape_colour = next(iter(shape_colours))
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if r == H - 1 and c == 0:
                    if out[r][c] != 0:
                        return None
                elif v == shape_colour:
                    if out[r][c] != corner:
                        return None
                elif v == 0:
                    if out[r][c] != 0:
                        return None
                else:
                    if out[r][c] != 0:
                        return None

    def solve(inp):
        H, W = len(inp), len(inp[0])
        corner = inp[H - 1][0]
        if corner == 0:
            return None
        shape_colours = {
            inp[r][c] for r in range(H) for c in range(W)
            if inp[r][c] != 0 and not (r == H - 1 and c == 0)
        }
        shape_colours.discard(corner)
        if len(shape_colours) != 1:
            return None
        shape_colour = next(iter(shape_colours))
        out = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if inp[r][c] == shape_colour:
                    out[r][c] = corner
        return out

    return solve


# ── Strategy: colour-5 as shape ───────────────────────────────────────────────

def _try_colour5_shape(task):
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        H, W = len(inp), len(inp[0])
        if len(out) != H or len(out[0]) != W:
            return None
        colours = {inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0}
        if 5 not in colours or len(colours) != 2:
            return None
        other = next(c for c in colours if c != 5)
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                expected = other if v == 5 else 0
                if out[r][c] != expected:
                    return None

    def solve(inp):
        H, W = len(inp), len(inp[0])
        colours = {inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0}
        if 5 not in colours or len(colours) != 2:
            return None
        other = next(c for c in colours if c != 5)
        out = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if inp[r][c] == 5:
                    out[r][c] = other
        return out

    return solve


# ── Strategy registry and dispatch ───────────────────────────────────────────

_STRATEGIES = [
    ("static_substitution", _try_static_substitution),
    ("fill_most_common",     _try_fill_most_common),
    ("most_common_stays",    _try_most_common_stays),
    ("concentric_reverse",   _try_concentric_reverse),
    ("concentric_rotate",    _try_concentric_rotate),
    ("corner_indicator",     _try_corner_indicator),
    ("colour5_shape",        _try_colour5_shape),
]


def _verify(solve_fn, task) -> bool:
    """Check that solve_fn reproduces every training output exactly."""
    for p in task["train"]:
        inp, out = _to_list(p["input"]), _to_list(p["output"])
        try:
            pred = solve_fn(inp)
        except Exception:
            return False
        if pred is None or pred != out:
            return False
    return True


def _detect(task) -> tuple[str, object] | None:
    """Return (strategy_name, solve_fn) for the first matching strategy, or None."""
    for name, try_fn in _STRATEGIES:
        try:
            solve_fn = try_fn(task)
        except Exception:
            solve_fn = None
        if solve_fn is not None and _verify(solve_fn, task):
            return name, solve_fn
    return None


def detect_colour_remap(task: dict) -> bool:
    return _detect(task) is not None


def solve_colour_remap(input_grid: list[list[int]], task: dict) -> list[list[int]] | None:
    result = _detect(task)
    if result is None:
        return None
    _, solve_fn = result
    inp = _to_list(input_grid)
    return solve_fn(inp)


def categorise_colour_remap(task: dict) -> list[str]:
    return COLOUR_REMAP_CATEGORIES if detect_colour_remap(task) else []
