"""
human_tree.py — Apply the human-derived decision tree to all 400 training tasks.

The tree was hand-built by examining tasks and identifying the key decision
points a human uses.  This script applies it to all 400 tasks and reports:
  - How many tasks fall into each leaf category
  - Which tasks are unclassified (tree needs new branches)

Usage:
    python scripts/human_tree.py
    python scripts/human_tree.py --show-unclassified
    python scripts/human_tree.py --task 007bbfb7    # trace one task
"""

import argparse
import json
import numpy as np
from collections import defaultdict
from itertools import permutations as _permutations
from pathlib import Path
from scipy import ndimage

TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_task(task_id: str) -> dict:
    raw = json.loads((TRAINING_DIR / f"{task_id}.json").read_text())
    return {
        "task_id": task_id,
        "train": [
            {"input":  np.array(p["input"],  dtype=np.uint8),
             "output": np.array(p["output"], dtype=np.uint8)}
            for p in raw["train"]
        ],
    }


# ── Feature detectors ─────────────────────────────────────────────────────────

def same_size(task):
    return all(p["input"].shape == p["output"].shape for p in task["train"])

def is_identity(task):
    """True iff input equals output exactly in every training pair."""
    return all(np.array_equal(p["input"], p["output"]) for p in task["train"])

def output_is_single_cell(task):
    return all(p["output"].shape == (1, 1) for p in task["train"])

def output_is_multiple_of_input(task):
    for p in task["train"]:
        hi, wi = p["input"].shape
        ho, wo = p["output"].shape
        if ho < hi * 2 or wo < wi * 2:
            return False
        if ho % hi != 0 or wo % wi != 0:
            return False
        if ho // hi != wo // wi:
            return False
    return True

def one_dimension_same(task):
    """Output matches input in exactly one dimension (same width XOR same height)."""
    results = []
    for p in task["train"]:
        hi, wi = p["input"].shape
        ho, wo = p["output"].shape
        same_h = (hi == ho)
        same_w = (wi == wo)
        if same_h and same_w:
            return False   # fully same size — handled elsewhere
        if not same_h and not same_w:
            return False   # completely different
        results.append(True)
    return bool(results)

def no_colours_lost(task):
    for p in task["train"]:
        ic = set(np.unique(p["input"])) - {0}
        oc = set(np.unique(p["output"])) - {0}
        if ic - oc:
            return False
    return True

def all_input_colours_in_output(task):
    """All non-zero input colours appear somewhere in the output."""
    return no_colours_lost(task)

def has_new_colours(task):
    for p in task["train"]:
        ic = set(np.unique(p["input"])) - {0}
        oc = set(np.unique(p["output"])) - {0}
        if oc - ic:
            return True
    return False

def input_preserved_in_output(task):
    """Every non-zero input cell is at the same position with the same colour in output."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return False
        mask = inp > 0
        if not np.all(out[mask] == inp[mask]):
            return False
    return True

def get_components(grid):
    result = []
    for colour in np.unique(grid):
        if colour == 0:
            continue
        labeled, n = ndimage.label(grid == colour)
        for i in range(1, n + 1):
            result.append((int(colour), labeled == i))
    return result

def n_shapes_in_input(task):
    return min(len(get_components(p["input"])) for p in task["train"])

def has_static_shape(task):
    for p in task["train"]:
        for ci, mi in get_components(p["input"]):
            for co, mo in get_components(p["output"]):
                if ci == co and np.array_equal(mi, mo):
                    return True
    return False

def has_grid_lines(grid):
    h, w = grid.shape
    full_rows = sum(1 for r in range(h)
                    if len(set(grid[r])) == 1 and grid[r, 0] != 0)
    full_cols = sum(1 for c in range(w)
                    if len(set(grid[:, c])) == 1 and grid[0, c] != 0)
    return full_rows >= 2 or full_cols >= 2

def has_grid_structure(task):
    return all(has_grid_lines(p["input"]) for p in task["train"])

def input_is_monochrome(task):
    for p in task["train"]:
        if len(set(np.unique(p["input"])) - {0}) != 1:
            return False
    return True

def input_cells_are_columns(task):
    for p in task["train"]:
        for c in range(p["input"].shape[1]):
            col = p["input"][:, c]
            nz = col[col != 0]
            if len(nz) == 0:
                continue
            if len(set(nz)) > 1:
                return False
            pos = np.where(col != 0)[0]
            if pos[-1] - pos[0] != len(pos) - 1:
                return False
    return True

def has_vertical_and_horizontal_grid(task):
    """Both full-height vertical bars AND full-width horizontal bars present."""
    for p in task["train"]:
        grid = p["input"]
        h, w = grid.shape
        has_v = any(len(set(grid[:, c])) == 1 and grid[0, c] != 0 for c in range(w))
        has_h = any(len(set(grid[r])) == 1 and grid[r, 0] != 0 for r in range(h))
        if not (has_v and has_h):
            return False
    return True

def one_vertical_bar_two_equal_halves(task):
    """Exactly one uniform-colour full-height column divides input into two equal halves."""
    for p in task["train"]:
        grid = p["input"]
        h, w = grid.shape
        bar_cols = [c for c in range(w) if len(set(grid[:, c])) == 1]
        if len(bar_cols) != 1:
            return False
        bc = bar_cols[0]
        if bc != (w - 1) // 2:   # bar must be at the centre
            return False
        left_w  = bc
        right_w = w - bc - 1
        if left_w != right_w:
            return False
    return True

def output_grows_in_free_dim(task):
    """In the one-dim-same case: True if the non-shared dimension grows (output > input)."""
    results = []
    for p in task["train"]:
        hi, wi = p["input"].shape
        ho, wo = p["output"].shape
        if hi == ho:   # height is shared → width is free
            results.append(wo > wi)
        elif wi == wo:  # width is shared → height is free
            results.append(ho > hi)
        else:
            return False
    return all(results) if results else False


def cells_fill_enclosed_interior(task):
    """True iff:
      - in every training pair, the zero-cells filled in the output are exactly
        those NOT reachable from the grid border (i.e. enclosed zeros), AND
      - at least one pair actually has enclosed zeros that get filled.

    This is the defining property of a genuine flood-fill-enclosed task.
    """
    any_fill = False

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return False

        rows, cols = inp.shape
        reachable = np.zeros((rows, cols), dtype=bool)

        # BFS from every border zero
        queue = []
        for r in range(rows):
            for c in (0, cols - 1):
                if inp[r, c] == 0 and not reachable[r, c]:
                    reachable[r, c] = True
                    queue.append((r, c))
        for c in range(cols):
            for r in (0, rows - 1):
                if inp[r, c] == 0 and not reachable[r, c]:
                    reachable[r, c] = True
                    queue.append((r, c))

        head = 0
        while head < len(queue):
            r, c = queue[head]; head += 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and inp[nr, nc] == 0:
                    reachable[nr, nc] = True
                    queue.append((nr, nc))

        interior = (inp == 0) & ~reachable          # zeros not reachable from border
        filled   = (inp == 0) & (out != 0)          # zeros that got coloured

        # Any non-zero cell that changed colour means this is NOT a pure zero-fill task
        recoloured = (inp != 0) & (out != 0) & (inp != out)
        if recoloured.any():
            return False

        if not np.array_equal(filled, interior):
            return False                             # mismatch → not a pure enclosed fill

        if interior.any():
            any_fill = True

    return any_fill   # must have at least one pair with genuine enclosed fill


def has_unique_colour_shape(task):
    """At least one shape whose colour appears in exactly one connected component in the input."""
    for p in task["train"]:
        comps = get_components(p["input"])
        colour_count = defaultdict(int)
        for c, _ in comps:
            colour_count[c] += 1
        if any(n == 1 for n in colour_count.values()):
            return True
    return False


# ── EXTRACT_UNIQUE_SHAPE sub-categories ───────────────────────────────────────

def is_colour_bands_uniform(task):
    """All rows uniform XOR all columns uniform; output = run-length-encoded colour sequence."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if np.any(inp == 0):
            return False
        oh, ow = out.shape
        if oh != 1 and ow != 1:
            return False
        H, W = inp.shape
        rows_uniform = all(len(set(inp[r].tolist())) == 1 for r in range(H))
        cols_uniform = all(len(set(inp[:, c].tolist())) == 1 for c in range(W))
        if not rows_uniform and not cols_uniform:
            return False
        if rows_uniform and ow == 1:
            seq = [int(inp[r, 0]) for r in range(H)]
        elif cols_uniform and oh == 1:
            seq = [int(inp[0, c]) for c in range(W)]
        else:
            return False
        rle = []
        for v in seq:
            if not rle or rle[-1] != v:
                rle.append(v)
        expected = (np.array(rle).reshape(-1, 1) if ow == 1
                    else np.array(rle).reshape(1, -1))
        if not np.array_equal(expected, out):
            return False
    return True


def is_zero_block_complete(task):
    """Input is mostly non-zero; all zeros form a single solid rectangle.
    Output has the same shape as that rectangle (completing the pattern)."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        zero_frac = np.sum(inp == 0) / inp.size
        if zero_frac > 0.25 or zero_frac == 0:
            return False
        zero_rows, zero_cols = np.where(inp == 0)
        rmin, cmin = zero_rows.min(), zero_cols.min()
        rmax, cmax = zero_rows.max(), zero_cols.max()
        block = inp[rmin:rmax + 1, cmin:cmax + 1]
        if not np.all(block == 0):
            return False          # zeros not in a solid rectangle
        if out.shape != block.shape:
            return False
    return True


def _is_hollow_rect_sparse(inp, colour):
    """True if colour forms exactly the border of a rectangle on a zero background."""
    rows, cols = np.where(inp == colour)
    if len(rows) == 0:
        return False
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    if rmax - rmin < 2 or cmax - cmin < 2:
        return False
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            on_border = (r == rmin or r == rmax or c == cmin or c == cmax)
            if on_border and inp[r, c] != colour:
                return False
            if not on_border and inp[r, c] != 0:
                return False
    return True


def is_largest_hollow_rect(task):
    """All non-zero components form hollow rectangular outlines on a sparse background.
    Output is a solid block whose colour equals the colour of the largest outline."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        colours = [int(c) for c in np.unique(inp) if c != 0]
        if len(colours) < 2:
            return False
        for colour in colours:
            if not _is_hollow_rect_sparse(inp, colour):
                return False
        # Output must be a uniform solid block of one colour
        out_vals = set(np.unique(out).tolist())
        if len(out_vals) != 1 or list(out_vals)[0] == 0:
            return False
        out_colour = int(list(out_vals)[0])
        # The output colour must be the colour of the largest outline
        areas = {}
        for colour in colours:
            rows, cols = np.where(inp == colour)
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            areas[colour] = (rmax - rmin + 1) * (cmax - cmin + 1)
        if out_colour != max(areas, key=areas.get):
            return False
    return True


def is_extract_rect_interior(task):
    """One colour in the input forms a hollow rectangular outline (border-only).
    Output = the cells inside that rectangle."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        colours = [int(c) for c in np.unique(inp) if c != 0]
        found = False
        for colour in colours:
            rows, cols = np.where(inp == colour)
            if len(rows) == 0:
                continue
            rmin, rmax = rows.min(), rows.max()
            cmin, cmax = cols.min(), cols.max()
            if rmax - rmin < 2 or cmax - cmin < 2:
                continue
            # All top/bottom border cells must be this colour
            if not all(inp[rmin, c] == colour for c in range(cmin, cmax + 1)):
                continue
            if not all(inp[rmax, c] == colour for c in range(cmin, cmax + 1)):
                continue
            if not all(inp[r, cmin] == colour for r in range(rmin, rmax + 1)):
                continue
            if not all(inp[r, cmax] == colour for r in range(rmin, rmax + 1)):
                continue
            # No interior cells may be this colour
            interior = inp[rmin + 1:rmax, cmin + 1:cmax]
            if np.any(interior == colour):
                continue
            if np.array_equal(interior, out):
                found = True
                break
        if not found:
            return False
    return True


def is_odd_cell_embedded(task):
    """Multiple spatial clusters; exactly one cluster per pair has exactly one cell of
    an intruder colour (different from the cluster's dominant colour).
    Output = bounding box of that cluster with the intruder replaced by the dominant colour."""
    intruder_colour = None
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        labeled, n = ndimage.label(inp != 0)
        if n < 2:
            return False
        odd_clusters = []
        for idx in range(1, n + 1):
            mask = labeled == idx
            vals = inp[mask]
            colours, counts = np.unique(vals, return_counts=True)
            if len(colours) <= 1:
                continue
            minority = [(int(c), int(cnt)) for c, cnt in zip(colours, counts) if cnt == 1]
            if len(minority) != 1:
                continue
            dominant = int(colours[counts.argmax()])
            foreign = minority[0][0]
            odd_clusters.append((idx, dominant, foreign))
        if len(odd_clusters) != 1:
            return False
        idx, dominant, foreign = odd_clusters[0]
        if intruder_colour is None:
            intruder_colour = foreign
        elif intruder_colour != foreign:
            return False
        mask = labeled == idx
        nz = np.argwhere(mask)
        rmin, cmin = nz.min(axis=0)
        rmax, cmax = nz.max(axis=0)
        bbox = inp[rmin:rmax + 1, cmin:cmax + 1].copy()
        bbox[bbox == foreign] = dominant
        if not np.array_equal(bbox, out):
            return False
    return True


def is_single_shape_sparse(task):
    """Sparse input (≥60% zeros), single spatially connected non-zero cluster.
    Output is smaller than input in both area and each dimension."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if np.sum(inp == 0) / inp.size < 0.60:
            return False
        nz = np.argwhere(inp != 0)
        if len(nz) == 0:
            return False
        struct8 = ndimage.generate_binary_structure(2, 2)
        _, n_comps = ndimage.label(inp != 0, structure=struct8)
        if n_comps != 1:
            return False
        # The largest 4-connected component must contain >50% of nonzero cells
        # (prevents spurious single-cluster from diagonally-touching separate blobs)
        labeled4, _ = ndimage.label(inp != 0)
        total_nz = int(np.sum(inp != 0))
        max_comp = max(int(np.sum(labeled4 == k)) for k in range(1, labeled4.max() + 1))
        if max_comp / total_nz < 0.40:
            return False
        if out.size >= inp.size:
            return False
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            return False
    return True


# ── EXTRACT_UNIQUE_SHAPE sub-categories (continued) ───────────────────────────

def _shape_pattern(grid, colour):
    """Return binary mask of colour cells in their minimal bounding box."""
    rows, cols = np.where(grid == colour)
    if len(rows) == 0:
        return None
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    return (grid[rmin:rmax+1, cmin:cmax+1] == colour)


def is_jigsaw_fill_rect(task):
    """Exactly two colours in input, each one 4-connected component.
    Together their cells exactly fill the output rectangle (no zeros in output).
    The shape pattern of each colour matches between input and output."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        colours = [int(c) for c in np.unique(inp) if c != 0]
        if len(colours) != 2:
            return False
        c0, c1 = colours[0], colours[1]
        for col in [c0, c1]:
            _, n = ndimage.label(inp == col)
            if n != 1:
                return False
        n0 = int(np.sum(inp == c0))
        n1 = int(np.sum(inp == c1))
        if n0 + n1 != out.size:
            return False
        if np.any(out == 0):
            return False
        if set(int(c) for c in np.unique(out)) != {c0, c1}:
            return False
        if int(np.sum(out == c0)) != n0 or int(np.sum(out == c1)) != n1:
            return False
        in_s0 = _shape_pattern(inp, c0)
        in_s1 = _shape_pattern(inp, c1)
        out_s0 = _shape_pattern(out, c0)
        out_s1 = _shape_pattern(out, c1)
        if any(s is None for s in [in_s0, in_s1, out_s0, out_s1]):
            return False
        if not np.array_equal(in_s0, out_s0) or not np.array_equal(in_s1, out_s1):
            return False
    return True


def _get_block_colours(inp):
    """If input is a uniform-block grid separated by all-zero rows and columns,
    return a flat list of block colours (one per block slot); else return None."""
    h, w = inp.shape
    zero_rows = {r for r in range(h) if np.all(inp[r] == 0)}
    zero_cols = {c for c in range(w) if np.all(inp[:, c] == 0)}
    if not zero_rows or not zero_cols:
        return None

    def _groups(n, zero_set):
        groups = []
        i = 0
        while i < n:
            if i not in zero_set:
                s = i
                while i < n and i not in zero_set:
                    i += 1
                groups.append((s, i - 1))
            else:
                i += 1
        return groups

    block_rows = _groups(h, zero_rows)
    block_cols = _groups(w, zero_cols)
    if len(block_rows) < 2 or len(block_cols) < 2:
        return None
    colours = []
    for rs, re in block_rows:
        for cs, ce in block_cols:
            block = inp[rs:re+1, cs:ce+1]
            uniq = np.unique(block)
            if len(uniq) != 1:
                return None
            colours.append(int(uniq[0]))
    return colours


def is_block_grid_rank(task):
    """Grid of uniform blocks separated by all-zero rows and columns.
    Count blocks per colour; discard the most-common colour;
    output = remaining colours as a single column, sorted by count descending."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        blocks = _get_block_colours(inp)
        if blocks is None:
            return False
        cols_arr, cnts_arr = np.unique(blocks, return_counts=True)
        colour_counts = dict(zip(cols_arr.tolist(), cnts_arr.tolist()))
        if len(colour_counts) < 2:
            return False
        most_common = max(colour_counts, key=colour_counts.get)
        remaining = {c: n for c, n in colour_counts.items() if c != most_common}
        if not remaining:
            return False
        sorted_remaining = sorted(remaining.items(), key=lambda x: -x[1])
        expected_colours = [c for c, _ in sorted_remaining]
        if out.shape != (len(expected_colours), 1):
            return False
        expected = np.array(expected_colours, dtype=np.uint8).reshape(-1, 1)
        if not np.array_equal(expected, out):
            return False
    return True


def is_largest_shape_output(task):
    """Multiple coloured shapes in sparse input.
    Output = bounding-box crop of the shape with the most cells."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        comps = get_components(inp)
        if len(comps) < 2:
            return False
        largest_colour, largest_mask = max(comps, key=lambda x: int(np.sum(x[1])))
        nz = np.argwhere(largest_mask)
        rmin, cmin = nz.min(axis=0)
        rmax, cmax = nz.max(axis=0)
        bbox = inp[rmin:rmax+1, cmin:cmax+1]
        if not np.array_equal(bbox, out):
            return False
    return True


def is_colour_order_by_size(task):
    """Sparse input with multiple colours.
    Output = K×1 column listing colours sorted by total cell count descending."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if out.shape[1] != 1:
            return False
        colours = [int(c) for c in np.unique(inp) if c != 0]
        if len(colours) < 2:
            return False
        colour_counts = sorted([(c, int(np.sum(inp == c))) for c in colours],
                               key=lambda x: -x[1])
        expected_colours = [c for c, _ in colour_counts]
        if len(expected_colours) != out.shape[0]:
            return False
        expected = np.array(expected_colours, dtype=np.uint8).reshape(-1, 1)
        if not np.array_equal(expected, out):
            return False
    return True


def is_colour_bars_by_count(task):
    """Dense input (no zeros). Output = H×K bar chart where H = max cell count,
    K = number of colours. Each column is one colour, filled from the top for
    count rows then zero. Columns sorted left-to-right by count descending."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if np.any(inp == 0):
            return False
        colours, counts = np.unique(inp, return_counts=True)
        order = np.argsort(-counts)
        sorted_colours = colours[order].tolist()
        sorted_counts = counts[order].tolist()
        max_count = sorted_counts[0]
        K = len(sorted_colours)
        if out.shape != (max_count, K):
            return False
        for col_idx, (colour, count) in enumerate(zip(sorted_colours, sorted_counts)):
            col = out[:, col_idx]
            expected = np.array([colour] * count + [0] * (max_count - count),
                                dtype=np.uint8)
            if not np.array_equal(col, expected):
                return False
    return True


def is_colour_bars_max_shapes(task):
    """Sparse input with multiple shapes. Only shapes with the maximum cell count
    appear in the output as solid bars. Bars sorted left-to-right by each shape's
    leftmost column. Output height = max count, width = number of max-count shapes."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        comps = get_components(inp)
        if len(comps) < 2:
            return False
        count_list = [(colour, mask, int(np.sum(mask))) for colour, mask in comps]
        max_count = max(c for _, _, c in count_list)
        max_comps = [(colour, mask) for colour, mask, cnt in count_list
                     if cnt == max_count]
        max_comps_sorted = sorted(max_comps,
                                  key=lambda item: int(np.where(item[1])[1].min()))
        K = len(max_comps_sorted)
        if out.shape != (max_count, K):
            return False
        for col_idx, (colour, _) in enumerate(max_comps_sorted):
            expected_col = np.full(max_count, colour, dtype=np.uint8)
            if not np.array_equal(out[:, col_idx], expected_col):
                return False
    return True


def is_block_count_x(task):
    """Input contains N non-overlapping 2×2 blocks of a single colour (N ≤ 5).
    Output is a 3×3 grid with exactly N ones placed at the X-pattern positions
    (top-left, top-right, centre, bottom-left, bottom-right) in that order."""
    X_ORDER = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        nz_vals = set(inp[inp != 0].tolist())
        if len(nz_vals) != 1:
            return False
        colour = list(nz_vals)[0]
        h, w = inp.shape
        covered = np.zeros((h, w), dtype=bool)
        n_blocks = 0
        for r in range(h - 1):
            for c in range(w - 1):
                if np.all(inp[r:r+2, c:c+2] == colour) and not covered[r, c]:
                    covered[r:r+2, c:c+2] = True
                    n_blocks += 1
        if not np.all(covered == (inp == colour)):
            return False
        if n_blocks == 0 or n_blocks > 5:
            return False
        if out.shape != (3, 3):
            return False
        expected = np.zeros((3, 3), dtype=np.uint8)
        for k in range(n_blocks):
            expected[X_ORDER[k]] = 1
        if not np.array_equal(expected, out):
            return False
    return True


# ── Geometric transforms ───────────────────────────────────────────────────────

def _consistent_transform(task, fn):
    """True iff applying fn to every training input produces the corresponding output exactly."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        try:
            t = fn(inp)
        except Exception:
            return False
        if t.shape != out.shape:
            return False
        if not np.array_equal(t, out):
            return False
    return True


def is_reflection(task):
    """True iff one consistent reflection (H, V, main-diagonal, anti-diagonal)
    maps every training input to its output."""
    return (
        _consistent_transform(task, np.flipud)                          # top ↔ bottom
        or _consistent_transform(task, np.fliplr)                      # left ↔ right
        or _consistent_transform(task, lambda g: g.T)                  # main diagonal
        or _consistent_transform(task, lambda g: np.flip(g.T, (0, 1)))# anti-diagonal
    )


def is_rotation(task):
    """True iff one consistent rotation (90°, 180°, 270° CCW) maps every
    training input to its output."""
    return (
        _consistent_transform(task, lambda g: np.rot90(g, 1))  # 90° CCW
        or _consistent_transform(task, lambda g: np.rot90(g, 2))  # 180°
        or _consistent_transform(task, lambda g: np.rot90(g, 3))  # 270° CCW
    )


# ── 2×2 tiled output from transforms ─────────────────────────────────────────

def _is_2x2_transform_tile(task, transforms):
    """True iff the output is a 2×2 tiling of the input using some consistent
    permutation of the 4 given transforms, for every training pair.

    Tries all 24 orderings of the 4 quadrant positions so the task doesn't
    need to use a specific layout.
    """
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        H, W = inp.shape
        if out.shape != (2 * H, 2 * W):
            return False
        quadrants = [out[r * H:(r + 1) * H, c * W:(c + 1) * W]
                     for r in range(2) for c in range(2)]
        matched = False
        for perm in _permutations(transforms):
            try:
                if all(np.array_equal(fn(inp), q)
                       for fn, q in zip(perm, quadrants)):
                    matched = True
                    break
            except Exception:
                pass
        if not matched:
            return False
    return True


_ROTATE_4_FNS = (
    lambda g: g,
    lambda g: np.rot90(g, 1),
    lambda g: np.rot90(g, 2),
    lambda g: np.rot90(g, 3),
)

_REFLECT_4_FNS = (
    lambda g: g,
    np.fliplr,
    np.flipud,
    lambda g: np.flip(g, (0, 1)),   # flip both axes (= 180° rotation)
)


def is_tile_rotate_4(task):
    """True iff the output is a 2×2 grid containing the input at 0°, 90°, 180°, 270°."""
    return _is_2x2_transform_tile(task, _ROTATE_4_FNS)


def is_tile_reflect_4(task):
    """True iff the output is a 2×2 grid containing the input, its left-right mirror,
    its top-bottom mirror, and both mirrors combined."""
    return _is_2x2_transform_tile(task, _REFLECT_4_FNS)


# ── Pixel zoom (each cell → solid block of its colour) ───────────────────────

def is_zoom(task):
    """True iff the output is a scaled-up version of the input where every cell
    becomes a solid rectangular block filled with that cell's colour.

    The scale factor must be uniform within a pair (same H and W multiplier)
    but may vary across pairs.
    """
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        hi, wi = inp.shape
        ho, wo = out.shape
        if ho % hi != 0 or wo % wi != 0:
            return False
        sh, sw = ho // hi, wo // wi
        if sh != sw or sh < 2:
            return False
        for i in range(hi):
            for j in range(wi):
                block = out[i * sh:(i + 1) * sh, j * sw:(j + 1) * sw]
                if not np.all(block == inp[i, j]):
                    return False
    return True


# ── Count-fill: N copies in first N block slots ───────────────────────────────

def is_count_fill(task):
    """True iff:
      - the output is a multiple of the input in both dimensions, and
      - N = number of nonzero cells in the input, and
      - the output is filled with exactly N copies of the input placed in the
        first N block positions (scanning left-to-right, top-to-bottom),
        with all remaining block positions empty.

    The output canvas size may vary across training pairs.
    """
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        H, W = inp.shape
        oh, ow = out.shape
        if oh % H != 0 or ow % W != 0:
            return False
        rows_b, cols_b = oh // H, ow // W
        N = int(np.sum(inp != 0))
        if N == 0 or N > rows_b * cols_b:
            return False
        for k in range(rows_b * cols_b):
            bi, bj = divmod(k, cols_b)
            block = out[bi * H:(bi + 1) * H, bj * W:(bj + 1) * W]
            if k < N:
                if not np.array_equal(block, inp):
                    return False
            else:
                if not np.all(block == 0):
                    return False
    return True


# ── Self-tiling ───────────────────────────────────────────────────────────────

def is_self_tile(task):
    """True iff the input is a square N×N grid and the output is N²×N²,
    where each N×N block of the output is either a copy of the input
    (when the corresponding input cell is non-zero) or all zeros
    (when the corresponding input cell is zero).

    Generalises naturally to any square size, though 3×3→9×9 is most common.
    """
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        H, W = inp.shape
        if H != W:
            return False                        # input must be square
        N = H
        if out.shape != (N * N, N * N):
            return False
        zeros_block = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            for j in range(N):
                block = out[i * N:(i + 1) * N, j * N:(j + 1) * N]
                expected = inp if inp[i, j] != 0 else zeros_block
                if not np.array_equal(block, expected):
                    return False
    return True


# ── Decision tree ─────────────────────────────────────────────────────────────

def classify(task, trace=False):

    def say(msg):
        if trace:
            print(f"  {msg}")

    # ── Single-cell output ────────────────────────────────────────────────────
    if output_is_single_cell(task):
        say("output_is_single_cell=YES  →  SINGLE_CELL_OUTPUT")
        return "SINGLE_CELL_OUTPUT"

    # ── Same size ─────────────────────────────────────────────────────────────
    if same_size(task):
        say("same_size=YES")

        if is_identity(task):
            say("is_identity=YES  →  IDENTITY")
            return "IDENTITY"

        if input_is_monochrome(task) and input_cells_are_columns(task) and has_new_colours(task):
            say("monochrome_columns + new_colours  →  COLOUR_BY_HEIGHT")
            return "COLOUR_BY_HEIGHT"

        if has_grid_structure(task):
            say("has_grid_structure=YES  →  COLOUR_BETWEEN_PAIRS")
            return "COLOUR_BETWEEN_PAIRS"

        if no_colours_lost(task):
            say("no_colours_lost=YES")

            if has_new_colours(task):
                say("has_new_colours=YES")
                if cells_fill_enclosed_interior(task):
                    say("cells_fill_enclosed_interior=YES  →  FILL_REGIONS")
                    return "FILL_REGIONS"
                say("cells_fill_enclosed_interior=NO  →  SAME_SIZE_NEW_COLOURS")
                return "SAME_SIZE_NEW_COLOURS"

            if is_reflection(task):
                say("is_reflection=YES  →  REFLECT")
                return "REFLECT"

            if is_rotation(task):
                say("is_rotation=YES  →  ROTATE")
                return "ROTATE"

            n = n_shapes_in_input(task)
            say(f"n_shapes={n}")

            if n == 1:
                say("→ FILL_WITH_SHAPE")
                return "FILL_WITH_SHAPE"
            if n > 1:
                if has_static_shape(task):
                    say("has_static_shape=YES  →  MOVE_TO_STATIC")
                    return "MOVE_TO_STATIC"
                else:
                    say("has_static_shape=NO  →  MOVE_PART")
                    return "MOVE_PART"

        say("no_colours_lost=NO")
        if not has_new_colours(task):
            say("no_new_colours  →  COLOUR_REMOVAL")
            return "COLOUR_REMOVAL"
        say("has_new_colours=YES  →  COLOUR_SUBSTITUTION")
        return "COLOUR_SUBSTITUTION"

    # ── Output is integer multiple of input ───────────────────────────────────
    if output_is_multiple_of_input(task):
        say("output_is_multiple=YES")

        if is_zoom(task):
            say("is_zoom=YES  →  ZOOM")
            return "ZOOM"

        if is_count_fill(task):
            say("is_count_fill=YES  →  COUNT_FILL")
            return "COUNT_FILL"

        if is_self_tile(task):
            say("is_self_tile=YES  →  SELF_TILE")
            return "SELF_TILE"

        if is_tile_rotate_4(task):
            say("is_tile_rotate_4=YES  →  TILE_ROTATE_4")
            return "TILE_ROTATE_4"

        if is_tile_reflect_4(task):
            say("is_tile_reflect_4=YES  →  TILE_REFLECT_4")
            return "TILE_REFLECT_4"

        say("→  TILE_ASSEMBLY")
        return "TILE_ASSEMBLY"

    # ── One dimension shared between input and output ─────────────────────────
    if one_dimension_same(task):
        say("one_dimension_same=YES")

        if has_vertical_and_horizontal_grid(task):
            say("has_vertical_and_horizontal_grid=YES  →  GRID_SELECT_ELEMENT")
            return "GRID_SELECT_ELEMENT"

        if one_vertical_bar_two_equal_halves(task):
            say("one_vertical_bar_two_equal_halves=YES  →  AND_HALVES")
            return "AND_HALVES"

        if is_colour_bars_by_count(task):
            say("is_colour_bars_by_count=YES  →  COLOUR_BARS_BY_COUNT")
            return "COLOUR_BARS_BY_COUNT"

        if output_grows_in_free_dim(task):
            say("output_grows_in_free_dim=YES  →  ONE_DIM_EXTEND")
            return "ONE_DIM_EXTEND"

        say("output_shrinks_in_free_dim  →  ONE_DIM_CROP")
        return "ONE_DIM_CROP"

    # ── Grids different size (catch-all) ──────────────────────────────────────
    say("different_size")

    if is_colour_bands_uniform(task):
        say("is_colour_bands_uniform=YES  →  COLOUR_BANDS")
        return "COLOUR_BANDS"

    if is_zero_block_complete(task):
        say("is_zero_block_complete=YES  →  COMPLETE_PATTERN")
        return "COMPLETE_PATTERN"

    if is_largest_hollow_rect(task):
        say("is_largest_hollow_rect=YES  →  LARGEST_OUTLINE")
        return "LARGEST_OUTLINE"

    if is_extract_rect_interior(task):
        say("is_extract_rect_interior=YES  →  EXTRACT_RECT_INTERIOR")
        return "EXTRACT_RECT_INTERIOR"

    if is_odd_cell_embedded(task):
        say("is_odd_cell_embedded=YES  →  ODD_CELL_RECOLOUR")
        return "ODD_CELL_RECOLOUR"

    if is_single_shape_sparse(task):
        say("is_single_shape_sparse=YES  →  SINGLE_SHAPE_EXTRACT")
        return "SINGLE_SHAPE_EXTRACT"

    if is_colour_bars_by_count(task):
        say("is_colour_bars_by_count=YES  →  COLOUR_BARS_BY_COUNT")
        return "COLOUR_BARS_BY_COUNT"

    if is_block_count_x(task):
        say("is_block_count_x=YES  →  BLOCK_COUNT_X")
        return "BLOCK_COUNT_X"

    if is_jigsaw_fill_rect(task):
        say("is_jigsaw_fill_rect=YES  →  JIGSAW_FILL_RECT")
        return "JIGSAW_FILL_RECT"

    if is_block_grid_rank(task):
        say("is_block_grid_rank=YES  →  BLOCK_GRID_RANK")
        return "BLOCK_GRID_RANK"

    if is_largest_shape_output(task):
        say("is_largest_shape_output=YES  →  LARGEST_SHAPE")
        return "LARGEST_SHAPE"

    if is_colour_order_by_size(task):
        say("is_colour_order_by_size=YES  →  COLOUR_ORDER_BY_SIZE")
        return "COLOUR_ORDER_BY_SIZE"

    if is_colour_bars_max_shapes(task):
        say("is_colour_bars_max_shapes=YES  →  COLOUR_BARS_MAX")
        return "COLOUR_BARS_MAX"

    if has_unique_colour_shape(task):
        say("has_unique_colour_shape=YES  →  EXTRACT_UNIQUE_SHAPE")
        return "EXTRACT_UNIQUE_SHAPE"

    if all_input_colours_in_output(task):
        say("all_input_colours_in_output=YES")
        n = n_shapes_in_input(task)
        say(f"n_shapes={n}")
        if n > 1 and not has_static_shape(task):
            say("→  OVERLAP_OR_COUNT_SHAPES")
            return "OVERLAP_OR_COUNT_SHAPES"
        if n <= 1:
            say("→  INPUT_FROM_OUTPUT_TRANSFORMS")
            return "INPUT_FROM_OUTPUT_TRANSFORMS"

    # Output is smaller than input in both dims → extract an embedded object
    if all(p["output"].shape[0] * p["output"].shape[1] <
           p["input"].shape[0] * p["input"].shape[1]
           for p in task["train"]):
        say("output_area<input_area  →  EXTRACT_OBJECT")
        return "EXTRACT_OBJECT"

    say("→ UNCLASSIFIED")
    return "UNCLASSIFIED"


# ── Main ──────────────────────────────────────────────────────────────────────

CLASSIFIED_LABELS = {
    "SINGLE_CELL_OUTPUT",
    "COLOUR_BY_HEIGHT", "COLOUR_BETWEEN_PAIRS",
    "FILL_REGIONS", "SAME_SIZE_NEW_COLOURS", "FILL_WITH_SHAPE",
    "IDENTITY",
    "REFLECT", "ROTATE",
    "TILE_ROTATE_4", "TILE_REFLECT_4", "SELF_TILE", "ZOOM", "COUNT_FILL",
    "MOVE_TO_STATIC", "MOVE_PART",
    "COLOUR_REMOVAL", "COLOUR_SUBSTITUTION",
    "TILE_ASSEMBLY",
    "GRID_SELECT_ELEMENT", "AND_HALVES",
    "ONE_DIM_EXTEND", "ONE_DIM_CROP",
    "COLOUR_BANDS", "COMPLETE_PATTERN", "LARGEST_OUTLINE",
    "EXTRACT_RECT_INTERIOR", "ODD_CELL_RECOLOUR", "SINGLE_SHAPE_EXTRACT",
    "BLOCK_COUNT_X", "JIGSAW_FILL_RECT", "BLOCK_GRID_RANK",
    "LARGEST_SHAPE", "COLOUR_ORDER_BY_SIZE", "COLOUR_BARS_MAX",
    "COLOUR_BARS_BY_COUNT",
    "EXTRACT_UNIQUE_SHAPE", "EXTRACT_OBJECT",
    "OVERLAP_OR_COUNT_SHAPES", "INPUT_FROM_OUTPUT_TRANSFORMS",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-unclassified", action="store_true")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()

    task_ids = sorted(p.stem for p in TRAINING_DIR.glob("*.json"))

    if args.task:
        task = load_task(args.task)
        print(f"Task: {args.task}")
        result = classify(task, trace=True)
        print(f"  Category: {result}")
        return

    print(f"Running decision tree on {len(task_ids)} tasks...\n")
    categories = defaultdict(list)

    for tid in task_ids:
        try:
            cat = classify(load_task(tid))
        except Exception as e:
            cat = f"ERROR: {e}"
        categories[cat].append(tid)

    print(f"{'Category':<40} {'Count':>6}  Sample tasks")
    print("-" * 85)
    for cat, tids in sorted(categories.items(), key=lambda x: -len(x[1])):
        sample = "  " + " ".join(tids[:4]) + ("..." if len(tids) > 4 else "")
        print(f"  {cat:<38} {len(tids):>6}{sample}")

    n_classified   = sum(len(v) for k, v in categories.items() if k in CLASSIFIED_LABELS)
    n_unclassified = len(task_ids) - n_classified
    print(f"\n  Classified   : {n_classified} / {len(task_ids)}")
    print(f"  Unclassified : {n_unclassified} / {len(task_ids)}")

    if args.show_unclassified:
        uncl = categories.get("UNCLASSIFIED", [])
        print(f"\nUnclassified ({len(uncl)}):")
        for tid in uncl:
            print(f"  {tid}")


if __name__ == "__main__":
    main()
