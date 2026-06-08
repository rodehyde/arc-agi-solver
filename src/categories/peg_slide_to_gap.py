"""
PEG_SLIDE_TO_GAP (3906de3d training)

Input: a region of 1s in the upper part of the grid (with internal 0-gaps),
and one or more vertical segments of 2s below/outside the 1-region.

Rule: for each column containing a vertical 2-segment, slide that segment
upward (preserving its length) so that its top aligns with the first 0 in that
column that appears immediately after the last 1 from the top. The original
2-segment is erased.

Detection: every training pair has 1-segments in the upper portion and 2-
segments below; the slide prediction matches the output exactly.
"""


def _first_gap_after_ones(col):
    """Return index of first 0 that appears after at least one 1, or None."""
    seen_1 = False
    for r, v in enumerate(col):
        if v == 1:
            seen_1 = True
        elif v == 0 and seen_1:
            return r
    return None


def _predict(inp):
    H, W = len(inp), len(inp[0])
    out = [list(row) for row in inp]

    for c in range(W):
        col = [inp[r][c] for r in range(H)]

        # Locate the 2-segment (contiguous run of 2s)
        seg_start = seg_end = None
        for r in range(H):
            if col[r] == 2:
                if seg_start is None:
                    seg_start = r
                seg_end = r
        if seg_start is None:
            continue
        seg_len = seg_end - seg_start + 1

        first_gap = _first_gap_after_ones(col)
        if first_gap is None:
            continue

        # Erase old segment
        for r in range(seg_start, seg_end + 1):
            out[r][c] = 0
        # Place at first_gap
        for r in range(first_gap, first_gap + seg_len):
            if r < H:
                out[r][c] = 2

    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
        if _predict(inp) != out:
            return False
    return True


def solve(inp):
    return _predict(inp)


def categorise(task):
    return detect(task)
