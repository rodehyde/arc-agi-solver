"""
solvers.py — Delta-driven hypothesis generation for ARC-AGI tasks.

Framework
---------
1. compute_delta()  — characterise what changed in one input/output pair
2. task_delta()     — aggregate delta across all training pairs
3. verify()         — check a solver against all training pairs
4. find_solver()    — try applicable primitives, return first that verifies

Each primitive is a tuple: (name, applies_fn, solve_fn)
  applies_fn(delta: dict) -> bool          quick pre-filter on delta profile
  solve_fn(task: dict) -> list | None      returns one np.ndarray per test
                                           pair, or None if it can't explain

Usage
-----
    python scripts/solvers.py                   # run on all 400 training tasks
    python scripts/solvers.py --task 08ed6ac7   # trace one task
"""

import argparse
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.categories.rectangle_from_corners import (
    detect_rectangle_from_corners, solve_rectangle_from_corners,
)
from src.categories.gap_bridge import detect_gap_bridge, solve_gap_bridge
from src.categories.separator_grid_cross_fill import (
    detect_separator_grid_cross_fill, solve_separator_grid_cross_fill,
)
from src.categories.bounding_box_fill import (
    detect_bounding_box_fill, solve_bounding_box_fill,
)
from src.categories.hole_fill_2x2 import detect_hole_fill_2x2, solve_hole_fill_2x2
from src.categories.colour_marker_cross import (
    detect_colour_marker_cross, solve_colour_marker_cross,
)
from src.categories.vertical_comb import detect_vertical_comb, solve_vertical_comb
from src.categories.separator_grid_diagonal_fill import (
    detect_separator_grid_diagonal_fill, solve_separator_grid_diagonal_fill,
)
from src.categories.border_encoded_scale import (
    detect_border_encoded_scale, solve_border_encoded_scale,
)
from src.categories.quadrant_reflect import (
    detect_quadrant_reflect, solve_quadrant_reflect,
)
from src.categories.self_tile import detect_self_tile, solve_self_tile
from src.categories.line_fill_by_colour import (
    detect_line_fill_by_colour, solve_line_fill_by_colour,
)
from src.categories.row_fill_meet_middle import (
    detect_row_fill_meet_middle, solve_row_fill_meet_middle,
)
from src.categories.connect_aligned_pairs import (
    detect_connect_aligned_pairs, solve_connect_aligned_pairs,
)
from src.categories.geometric_transforms import solve_geometric_transform
from src.categories.logical_ops import solve_logical_op
from src.categories.colour_remap import solve_colour_remap
from src.categories.quadrant_mirror import solve_quadrant_mirror
from src.categories.tiling import solve_tile_fill

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
        "test": [
            {"input":  np.array(p["input"],  dtype=np.uint8),
             "output": (np.array(p["output"], dtype=np.uint8)
                        if "output" in p else None)}
            for p in raw["test"]
        ],
    }


# ── Delta computation ─────────────────────────────────────────────────────────

def compute_delta(inp: np.ndarray, out: np.ndarray) -> dict:
    """Characterise what changed between one same-size input/output pair."""
    return {
        "zeros_gained": int(np.sum((inp == 0) & (out != 0))),  # 0 → colour (filling)
        "zeros_lost":   int(np.sum((inp != 0) & (out == 0))),  # colour → 0 (erasure)
        "recoloured":   int(np.sum((inp != 0) & (out != 0) & (inp != out))),  # colour → different colour
        "unchanged":    int(np.sum(inp == out)),
        "new_colours":  sorted(set(out[out != 0].tolist()) - set(inp[inp != 0].tolist())),
        "lost_colours": sorted(set(inp[inp != 0].tolist()) - set(out[out != 0].tolist())),
        "total_cells":  inp.size,
    }


def task_delta(task: dict) -> dict:
    """Aggregate delta across all training pairs (sum counts, union colour sets).

    Skips pairs where input and output have different shapes — delta is only
    meaningful for same-size grids.  Returns all-zero counts for different-size tasks.
    """
    totals: dict = {"zeros_gained": 0, "zeros_lost": 0, "recoloured": 0,
                    "unchanged": 0, "total_cells": 0}
    new_c, lost_c = set(), set()
    for p in task["train"]:
        if p["input"].shape != p["output"].shape:
            continue  # delta undefined for different-size pairs
        d = compute_delta(p["input"], p["output"])
        for k in ("zeros_gained", "zeros_lost", "recoloured", "unchanged", "total_cells"):
            totals[k] += d[k]
        new_c.update(d["new_colours"])
        lost_c.update(d["lost_colours"])
    totals["new_colours"]  = sorted(new_c)
    totals["lost_colours"] = sorted(lost_c)
    return dict(totals)


# ── Verification ──────────────────────────────────────────────────────────────

def verify(solve_fn, task: dict) -> bool:
    """Return True iff solve_fn reproduces every training output exactly.

    Passes each training input through the solver (treating training pairs
    as if they were test pairs) and checks the output matches exactly.
    """
    check_task = {
        **task,
        "test": [{"input": p["input"]} for p in task["train"]],
    }
    try:
        preds = solve_fn(check_task)
    except Exception:
        return False
    if not preds:
        return False
    return all(
        pred is not None and np.array_equal(pred, pair["output"])
        for pred, pair in zip(preds, task["train"])
    )


# ── Primitives ────────────────────────────────────────────────────────────────
#
# Convention:
#   _applies_X(delta) -> bool     pre-filter: does this solver type make sense?
#   _solve_X(task)    -> list | None   full solver; None if it can't explain task
#
# The applies filter is cheap — it just looks at the delta profile.
# The solve function does the actual work and returns None on failure.

# ── COLOUR_BY_HEIGHT ──────────────────────────────────────────────────────────

def _applies_colour_by_height(d: dict) -> bool:
    """Only try this solver when cells are recoloured but nothing is gained or lost."""
    return (d["zeros_gained"] == 0
            and d["zeros_lost"] == 0
            and d["recoloured"] > 0
            and len(d["new_colours"]) > 0)


def _solve_colour_by_height(task: dict):
    """Columns are ranked by height (tallest=rank 1, shortest=rank N) and each
    rank is assigned a consistent colour learned from the training pairs.

    Returns None if the rank→colour mapping is inconsistent or incomplete.
    """
    rank_to_colour: dict[int, int] = {}

    for p in task["train"]:
        inp, out = p["input"], p["output"]

        # Collect (column_index, height) for every non-empty column
        col_heights = [(c, int(np.sum(inp[:, c] != 0)))
                       for c in range(inp.shape[1])
                       if np.any(inp[:, c] != 0)]
        if not col_heights:
            return None

        # Rank columns tallest-first (rank 1 = tallest)
        col_heights.sort(key=lambda x: -x[1])

        for rank, (c, _) in enumerate(col_heights, start=1):
            nz_out = out[:, c][out[:, c] != 0]
            if len(nz_out) == 0:
                return None
            if not np.all(nz_out == nz_out[0]):
                return None  # column has mixed colours in output
            colour = int(nz_out[0])
            if rank in rank_to_colour and rank_to_colour[rank] != colour:
                return None  # contradictory rank→colour across pairs
            rank_to_colour[rank] = colour

    if not rank_to_colour:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = np.zeros_like(inp)

        col_heights = [(c, int(np.sum(inp[:, c] != 0)))
                       for c in range(inp.shape[1])
                       if np.any(inp[:, c] != 0)]
        if not col_heights:
            results.append(out)
            continue

        col_heights.sort(key=lambda x: -x[1])

        for rank, (c, _) in enumerate(col_heights, start=1):
            if rank not in rank_to_colour:
                return None  # more columns in test than seen in training
            out[inp[:, c] != 0, c] = rank_to_colour[rank]

        results.append(out)

    return results


# ── FLOOD_FILL_ENCLOSED ───────────────────────────────────────────────────────

def _applies_flood_fill_enclosed(d: dict) -> bool:
    """Only try when zero-cells are filled (gained > 0) with exactly one new colour,
    no cells are erased, and no existing colours change."""
    return (d["zeros_gained"] > 0
            and d["zeros_lost"] == 0
            and d["recoloured"] == 0
            and len(d["new_colours"]) == 1)


def _flood_fill_interior(grid: np.ndarray) -> np.ndarray:
    """Return a boolean mask of cells that are 0 AND not reachable from any border."""
    rows, cols = grid.shape
    reachable = np.zeros((rows, cols), dtype=bool)

    # BFS from every border 0
    queue = []
    for r in range(rows):
        for c in (0, cols - 1):
            if grid[r, c] == 0 and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if grid[r, c] == 0 and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))

    head = 0
    while head < len(queue):
        r, c = queue[head]; head += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not reachable[nr, nc] and grid[nr, nc] == 0:
                reachable[nr, nc] = True
                queue.append((nr, nc))

    # Interior = 0 cells that were never reached
    return (grid == 0) & ~reachable


def _solve_flood_fill_enclosed(task: dict):
    """Fill all interior (enclosed) 0-regions with the single new colour learned
    from the training pairs.

    Returns None if:
    - the fill colour is inconsistent across training pairs
    - an interior mask doesn't match the gained-zero cells exactly
    """
    fill_colour: int | None = None

    for p in task["train"]:
        inp, out = p["input"], p["output"]

        interior = _flood_fill_interior(inp)

        # All filled cells in output must come from the interior mask
        filled = (inp == 0) & (out != 0)
        if not np.array_equal(filled, interior):
            return None  # some filled cells are not interior, or missed some interior

        # Colour must be uniform across all filled cells
        colours = set(out[interior].tolist())
        if len(colours) != 1:
            return None
        colour = colours.pop()

        if fill_colour is None:
            fill_colour = colour
        elif fill_colour != colour:
            return None  # colour inconsistent across pairs

    if fill_colour is None:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = inp.copy()
        interior = _flood_fill_interior(inp)
        out[interior] = fill_colour
        results.append(out)

    return results


# ── EXPAND_CROSS ──────────────────────────────────────────────────────────────
#
# Each small cross (centre colour C, arm colour A at distance 1 in all 4
# cardinal directions) expands into a larger star pattern:
#   A fills a plus-sign extending 2 steps in each of the 4 cardinal directions
#   C fills the 8 diagonal positions: 4 at distance 1, 4 at distance 2
#
# Example task: 0962bcdd

def _find_crosses(grid: np.ndarray) -> list:
    """Return [(centre_colour, arm_colour, row, col)] for every cross in grid.

    A cross is a non-zero centre cell whose four cardinal neighbours all share
    the same non-zero colour different from the centre.
    """
    rows, cols = grid.shape
    crosses = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            centre = int(grid[r, c])
            if centre == 0:
                continue
            arms = [int(grid[r-1, c]), int(grid[r+1, c]),
                    int(grid[r, c-1]), int(grid[r, c+1])]
            if all(a != 0 and a != centre for a in arms) and len(set(arms)) == 1:
                crosses.append((centre, arms[0], r, c))
    return crosses


def _applies_expand_cross(d: dict) -> bool:
    """Cells are gained but no new colours appear — expansion uses only existing colours."""
    return (d["zeros_gained"] > 0
            and d["zeros_lost"] == 0
            and d["recoloured"] == 0
            and len(d["new_colours"]) == 0)


_CROSS_CARDINAL = [(-2, 0), (-1, 0), (1, 0), (2, 0),
                   (0, -2), (0, -1), (0, 1), (0, 2)]
_CROSS_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                   (-2, -2), (-2, 2), (2, -2), (2, 2)]


def _expand_crosses(inp: np.ndarray):
    """Apply the expand-cross rule to a single grid.  Returns None if no crosses found."""
    crosses = _find_crosses(inp)
    if not crosses:
        return None
    out = inp.copy()
    rows, cols = inp.shape
    for centre_c, arm_c, r, c in crosses:
        for dr, dc in _CROSS_CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr, nc] = arm_c
        for dr, dc in _CROSS_DIAGONAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr, nc] = centre_c
    return out


def _solve_expand_cross(task: dict):
    results = []
    for tp in task["test"]:
        pred = _expand_crosses(tp["input"])
        if pred is None:
            return None
        results.append(pred)
    return results


# ── COLOUR_HALO ───────────────────────────────────────────────────────────────
#
# Each non-zero cell of colour S gains a halo of colour H in specific directions
# (cardinal and/or diagonal) that are empty (0) in the input.  The mapping
# S → (direction_type, H) is learned from training pairs.
#
# Example tasks: 0ca9ddb6 (c1→c7 cardinal halo, c2→c4 diagonal halo)

_CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _applies_colour_halo(d: dict) -> bool:
    """Cells gained; new colours appear; nothing erased or recoloured."""
    return (d["zeros_gained"] > 0
            and d["zeros_lost"] == 0
            and d["recoloured"] == 0
            and len(d["new_colours"]) > 0)


def _solve_colour_halo(task: dict):
    cardinal_halo: dict[int, int] = {}   # source colour → halo colour
    diagonal_halo: dict[int, int] = {}

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return None
        rows, cols = inp.shape

        for r in range(rows):
            for c in range(cols):
                src = int(inp[r, c])
                if src == 0:
                    continue

                for dr, dc in _CARDINAL:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and inp[nr, nc] == 0 and out[nr, nc] != 0:
                        h = int(out[nr, nc])
                        if src in cardinal_halo and cardinal_halo[src] != h:
                            return None  # contradictory mapping
                        cardinal_halo[src] = h

                for dr, dc in _DIAGONAL:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and inp[nr, nc] == 0 and out[nr, nc] != 0:
                        h = int(out[nr, nc])
                        if src in diagonal_halo and diagonal_halo[src] != h:
                            return None
                        diagonal_halo[src] = h

    if not cardinal_halo and not diagonal_halo:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = inp.copy()
        rows, cols = inp.shape

        for r in range(rows):
            for c in range(cols):
                src = int(inp[r, c])
                if src == 0:
                    continue

                if src in cardinal_halo:
                    for dr, dc in _CARDINAL:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and inp[nr, nc] == 0:
                            out[nr, nc] = cardinal_halo[src]

                if src in diagonal_halo:
                    for dr, dc in _DIAGONAL:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and inp[nr, nc] == 0:
                            out[nr, nc] = diagonal_halo[src]

        results.append(out)
    return results


# ── COLOUR_SUBSTITUTION ───────────────────────────────────────────────────────
#
# Every non-zero colour maps to a different non-zero colour; zeros are
# unchanged.  The mapping is learned from training pairs.  All training pairs
# must produce the same consistent mapping.
#
# Covers tasks in COLOUR_SUBSTITUTION and SAME_SIZE_NEW_COLOURS categories.

def _applies_colour_subst(d: dict) -> bool:
    """Cells are recoloured but none gained or lost."""
    return (d["zeros_gained"] == 0
            and d["zeros_lost"] == 0
            and d["recoloured"] > 0)


def _solve_colour_subst(task: dict):
    mapping: dict[int, int] = {}

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return None
        for src, dst in zip(inp.flat, out.flat):
            src, dst = int(src), int(dst)
            if src == dst:
                continue
            if src == 0 or dst == 0:
                return None  # zero ↔ non-zero not allowed here
            if src in mapping and mapping[src] != dst:
                return None  # contradictory mapping
            mapping[src] = dst

    if not mapping:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = inp.copy()
        rows, cols = inp.shape
        for r in range(rows):
            for c in range(cols):
                v = int(inp[r, c])
                if v in mapping:
                    out[r, c] = mapping[v]
        results.append(out)
    return results


# ── COLOUR_REMOVAL ────────────────────────────────────────────────────────────
#
# One or more colours in the input are set to 0 in the output; all other
# cells are unchanged.  The set of removed colours is learned from training.
#
# Covers tasks in COLOUR_REMOVAL category.

def _applies_colour_removal(d: dict) -> bool:
    """Cells lost but none gained or recoloured."""
    return (d["zeros_gained"] == 0
            and d["zeros_lost"] > 0
            and d["recoloured"] == 0
            and len(d["lost_colours"]) > 0)


def _solve_colour_removal(task: dict):
    removed: set[int] | None = None

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return None
        # Colours that are non-zero in input but 0 in output
        pair_removed = set(int(inp[r, c])
                           for r in range(inp.shape[0])
                           for c in range(inp.shape[1])
                           if inp[r, c] != 0 and out[r, c] == 0)
        if removed is None:
            removed = pair_removed
        elif removed != pair_removed:
            return None  # removed set differs across pairs

    if not removed:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = inp.copy()
        for colour in removed:
            out[out == colour] = 0
        results.append(out)
    return results


# ── UNIFORM_ROW_MARK ─────────────────────────────────────────────────────────
#
# Each row that is uniform (all non-zero cells share the same colour) is
# replaced with a fixed mark colour; non-uniform rows become all-zero.
# The mark colour is learned from training pairs (it is the single new colour
# that appears in the output).
#
# Example task: 25d8a9c8

def _applies_uniform_row_mark(d: dict) -> bool:
    """Rows are both zeroed (non-uniform) and recoloured (uniform); one new colour."""
    return (d["zeros_gained"] == 0
            and d["zeros_lost"] > 0
            and d["recoloured"] > 0
            and len(d["new_colours"]) == 1)


def _solve_uniform_row_mark(task: dict):
    d = task_delta(task)
    mark = d["new_colours"][0]

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return None
        for r in range(inp.shape[0]):
            nz = inp[r][inp[r] != 0]
            uniform = len(nz) > 0 and len(set(nz.tolist())) == 1
            expected = np.full(inp.shape[1], mark, dtype=np.uint8) if uniform else np.zeros(inp.shape[1], dtype=np.uint8)
            if not np.array_equal(out[r], expected):
                return None  # doesn't fit row-based uniform rule

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = np.zeros_like(inp)
        for r in range(inp.shape[0]):
            nz = inp[r][inp[r] != 0]
            if len(nz) > 0 and len(set(nz.tolist())) == 1:
                out[r] = mark
        results.append(out)
    return results


# ── MIRROR_AT_MARKER ──────────────────────────────────────────────────────────
#
# The input has two colours: a main shape (M) and a marker line (a short row
# or column of cells adjacent to one edge of M).  The output:
#   1. Fills all background 0-cells with a fill colour (learned from training).
#   2. Reflects M across the mirror plane defined by the marker line.
#   3. Replaces the marker cells with M (they coincide with the inner reflected
#      positions).
#
# Example task: 2bcee788

def _applies_mirror_at_marker(d: dict) -> bool:
    """Large zero-gain (background fill) + recoloured (marker→M) + one new colour + one lost colour."""
    return (d["zeros_gained"] > 0
            and d["zeros_lost"] == 0
            and d["recoloured"] > 0
            and len(d["new_colours"]) == 1
            and len(d["lost_colours"]) == 1)


def _mirror_one(inp: np.ndarray, marker_colour: int,
                main_colour: int, fill_colour: int):
    """Apply mirror-at-marker to a single grid.  Returns predicted output or None."""
    rows, cols = inp.shape

    marker_cells = [(r, c) for r in range(rows) for c in range(cols)
                    if int(inp[r, c]) == marker_colour]
    main_cells   = [(r, c) for r in range(rows) for c in range(cols)
                    if int(inp[r, c]) == main_colour]
    if not marker_cells or not main_cells:
        return None

    m_rs = set(r for r, c in marker_cells)
    m_cs = set(c for r, c in marker_cells)
    s_rs = set(r for r, c in main_cells)
    s_cs = set(c for r, c in main_cells)

    reflected = None

    # Try horizontal axis (marker occupies a single row band)
    if len(m_rs) == 1:
        r_m = next(iter(m_rs))
        if max(s_rs) < r_m:          # shape is above marker
            mirror = r_m - 0.5
        elif min(s_rs) > r_m:        # shape is below marker
            mirror = r_m + 0.5
        else:
            mirror = None
        if mirror is not None:
            reflected = [(int(2 * mirror - r), c) for r, c in main_cells]

    # Try vertical axis (marker occupies a single column band)
    if reflected is None and len(m_cs) == 1:
        c_m = next(iter(m_cs))
        if max(s_cs) < c_m:          # shape is left of marker
            mirror = c_m - 0.5
        elif min(s_cs) > c_m:        # shape is right of marker
            mirror = c_m + 0.5
        else:
            mirror = None
        if mirror is not None:
            reflected = [(r, int(2 * mirror - c)) for r, c in main_cells]

    if reflected is None:
        return None

    out = np.full((rows, cols), fill_colour, dtype=np.uint8)
    for r, c in main_cells:
        if 0 <= r < rows and 0 <= c < cols:
            out[r, c] = main_colour
    for r, c in reflected:
        if 0 <= r < rows and 0 <= c < cols:
            out[r, c] = main_colour
    return out


def _solve_mirror_at_marker(task: dict):
    d = task_delta(task)
    fill_colour   = d["new_colours"][0]
    marker_colour = d["lost_colours"][0]

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        colours = set(int(v) for v in inp.flat if v != 0)
        colours.discard(marker_colour)
        if len(colours) != 1:
            return None
        main_colour = colours.pop()
        pred = _mirror_one(inp, marker_colour, main_colour, fill_colour)
        if pred is None:
            return None
        results.append(pred)
    return results


# ── SHIFT_DOWN_ONE ────────────────────────────────────────────────────────────
#
# Every cell shifts down by one row; the top row becomes all zeros.
# Works when the bottom row is already all zeros (no cells fall off the grid).
#
# Example task: 25ff71a9

def _applies_shift_down_one(d: dict) -> bool:
    """Cells both gained and lost with no recolouring and no colour change."""
    return (d["zeros_gained"] > 0
            and d["zeros_lost"] > 0
            and d["recoloured"] == 0
            and len(d["new_colours"]) == 0
            and len(d["lost_colours"]) == 0)


def _solve_shift_down_one(task: dict):
    """Shift all cells down by one row; top row → zeros.  Verify on training first."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        if inp.shape != out.shape:
            return None
        expected = np.zeros_like(inp)
        expected[1:] = inp[:-1]
        if not np.array_equal(expected, out):
            return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        out = np.zeros_like(inp)
        out[1:] = inp[:-1]
        results.append(out)
    return results


# ── CROP_BOUNDING_BOX ─────────────────────────────────────────────────────────
#
# The output is the tightest bounding box around all non-zero cells in the input.
# Input and output have different sizes so task_delta returns all zeros (null delta).
#
# Example task: 1cf80156

def _applies_crop_bounding_box(d: dict) -> bool:
    """Null delta — all training pairs are size-changing (task_delta skips them)."""
    return d["total_cells"] == 0


def _solve_crop_bounding_box(task: dict):
    """Crop the input to the bounding box of non-zero cells."""
    for p in task["train"]:
        inp, out = p["input"], p["output"]
        rows_nz = np.any(inp != 0, axis=1)
        cols_nz = np.any(inp != 0, axis=0)
        if not np.any(rows_nz) or not np.any(cols_nz):
            return None
        r0, r1 = int(np.where(rows_nz)[0][0]),  int(np.where(rows_nz)[0][-1])
        c0, c1 = int(np.where(cols_nz)[0][0]),  int(np.where(cols_nz)[0][-1])
        cropped = inp[r0:r1+1, c0:c1+1]
        if not np.array_equal(cropped, out):
            return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        rows_nz = np.any(inp != 0, axis=1)
        cols_nz = np.any(inp != 0, axis=0)
        if not np.any(rows_nz) or not np.any(cols_nz):
            return None
        r0, r1 = int(np.where(rows_nz)[0][0]),  int(np.where(rows_nz)[0][-1])
        c0, c1 = int(np.where(cols_nz)[0][0]),  int(np.where(cols_nz)[0][-1])
        results.append(inp[r0:r1+1, c0:c1+1].copy())
    return results


# ── LOGICAL_AND_TWO_HALVES ────────────────────────────────────────────────────
#
# The input is split by a separator column (or row) of a single uniform colour
# into two equal halves.  A non-zero cell in the output marks positions where
# BOTH halves have non-zero cells.  The mark colour is learned from training.
# Input and output have different sizes so task_delta returns all zeros.
#
# Example task: 0520fde7

def _split_by_separator(inp: np.ndarray):
    """Return (left_half, right_half) if a uniform non-zero column splits the grid
    into two equal halves.  Also checks rows.  Returns None if not found."""
    rows, cols = inp.shape

    for c in range(cols):
        col = inp[:, c]
        if len(set(col.tolist())) == 1 and col[0] != 0:
            left  = inp[:, :c]
            right = inp[:, c+1:]
            if left.shape == right.shape and left.size > 0:
                return left, right

    for r in range(rows):
        row = inp[r, :]
        if len(set(row.tolist())) == 1 and row[0] != 0:
            top    = inp[:r, :]
            bottom = inp[r+1:, :]
            if top.shape == bottom.shape and top.size > 0:
                return top, bottom

    return None


def _applies_logical_and(d: dict) -> bool:
    """Null delta — all training pairs are size-changing."""
    return d["total_cells"] == 0


def _solve_logical_and(task: dict):
    """Mark positions where both halves are non-zero with the learned mark colour."""
    mark_colour = None

    for p in task["train"]:
        inp, out = p["input"], p["output"]
        halves = _split_by_separator(inp)
        if halves is None:
            return None
        left, right = halves
        if left.shape != out.shape:
            return None
        both_nz = (left != 0) & (right != 0)
        if not np.array_equal(both_nz, out != 0):
            return None
        colours = set(out[out != 0].tolist())
        if len(colours) != 1:
            return None
        mc = colours.pop()
        if mark_colour is None:
            mark_colour = mc
        elif mark_colour != mc:
            return None

    if mark_colour is None:
        return None

    results = []
    for tp in task["test"]:
        inp = tp["input"]
        halves = _split_by_separator(inp)
        if halves is None:
            return None
        left, right = halves
        out = np.zeros_like(left)
        out[(left != 0) & (right != 0)] = mark_colour
        results.append(out)
    return results


# ── Wrappers for category-module solvers ──────────────────────────────────────
#
# Category modules expose:
#   detect_*(task: dict) -> bool           — verifies rule on training pairs
#   solve_*(input_grid: list[list[int]])   — applies rule to one grid
#
# Here we wrap them into the (applies_fn, solve_fn) convention used by
# find_solver().  applies_fn returns True (detect is expensive; verify() does
# the real check).  solve_fn converts numpy <-> list at the boundary.

def _make_category_solver(solve_fn):
    """Return a _solve_X function that wraps a category-module solve_* function."""
    def _solve(task: dict):
        results = []
        for tp in task["test"]:
            inp = tp["input"]
            inp_list = inp.tolist() if hasattr(inp, "tolist") else inp
            out = solve_fn(inp_list)
            if out is None:
                return None
            results.append(np.array(out, dtype=np.uint8))
        return results
    return _solve


def _make_task_solver(solve_fn):
    """Variant of _make_category_solver for solvers that need the full task context
    (e.g. to infer colours from training pairs). solve_fn signature: (inp, task)."""
    def _solve(task: dict):
        results = []
        for tp in task["test"]:
            inp = tp["input"]
            inp_list = inp.tolist() if hasattr(inp, "tolist") else inp
            out = solve_fn(inp_list, task)
            if out is None:
                return None
            results.append(np.array(out, dtype=np.uint8))
        return results
    return _solve


_solve_rectangle_from_corners   = _make_category_solver(solve_rectangle_from_corners)
_solve_gap_bridge               = _make_category_solver(solve_gap_bridge)
_solve_separator_grid_cross     = _make_category_solver(solve_separator_grid_cross_fill)
_solve_bounding_box_fill        = _make_category_solver(solve_bounding_box_fill)
_solve_hole_fill_2x2            = _make_category_solver(solve_hole_fill_2x2)
_solve_colour_marker_cross      = _make_category_solver(solve_colour_marker_cross)
_solve_vertical_comb            = _make_category_solver(solve_vertical_comb)
_solve_separator_grid_diagonal  = _make_category_solver(solve_separator_grid_diagonal_fill)
_solve_border_encoded_scale     = _make_category_solver(solve_border_encoded_scale)
_solve_quadrant_reflect         = _make_category_solver(solve_quadrant_reflect)
_solve_self_tile                = _make_category_solver(solve_self_tile)
_solve_line_fill_by_colour      = _make_category_solver(solve_line_fill_by_colour)
_solve_row_fill_meet_middle     = _make_category_solver(solve_row_fill_meet_middle)
_solve_connect_aligned_pairs    = _make_task_solver(solve_connect_aligned_pairs)
_solve_quadrant_mirror          = _make_category_solver(solve_quadrant_mirror)
_solve_geometric_transform      = _make_task_solver(solve_geometric_transform)
_solve_logical_op               = _make_task_solver(solve_logical_op)
_solve_colour_remap             = _make_task_solver(solve_colour_remap)
_solve_tile_fill                = _make_category_solver(solve_tile_fill)


# ── Batch-1 task-specific solvers ─────────────────────────────────────────────
#
# All functions below take inp as a list-of-lists and return list-of-lists.
# Wrapped via _make_category_solver() so verify() gates them properly.

def _solve_fn_diagonal_tile(inp):
    """Anti-diagonal tiling: out[r][c] = seq[(r+c) % period]."""
    H, W = len(inp), len(inp[0])
    diags: dict[int, int] = {}
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0:
                d = r + c
                if d in diags and diags[d] != v:
                    return None
                diags[d] = v
    if not diags:
        return None
    n_vals = len(set(diags.values()))
    # Find minimum period: smallest P >= n_vals with no conflicts and full coverage
    period = None
    for P in range(n_vals, H + W + 1):
        seq_map: dict[int, int] = {}
        conflict = False
        for d, v in diags.items():
            m = d % P
            if m in seq_map and seq_map[m] != v:
                conflict = True; break
            seq_map[m] = v
        if not conflict and len(seq_map) == P:
            period = P
            break
    if period is None:
        return None
    seq_map = {d % period: v for d, v in diags.items()}
    return [[seq_map[(r + c) % period] for c in range(W)] for r in range(H)]


def _solve_fn_repeating_stripes(inp):
    """Two marker cells define alternating stripes (column or row) extending from markers."""
    H, W = len(inp), len(inp[0])
    markers = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != 0]
    if len(markers) != 2:
        return None
    m1, m2 = markers[0], markers[1]
    r1, c1, v1 = m1
    r2, c2, v2 = m2
    row_gap = abs(r2 - r1)
    col_gap = abs(c2 - c1)
    out = [[0] * W for _ in range(H)]
    if 0 < col_gap <= row_gap:
        # Column stripes: markers define which columns to fill
        if c1 > c2:
            c1, c2, v1, v2 = c2, c1, v2, v1
        gap = c2 - c1
        period = 2 * gap
        for c in range(c1, W):
            offset = c - c1
            if offset % period == 0:
                for r in range(H): out[r][c] = v1
            elif offset % period == gap:
                for r in range(H): out[r][c] = v2
    else:
        # Row stripes: markers define which rows to fill
        if r1 > r2:
            r1, r2, v1, v2 = r2, r1, v2, v1
        gap = r2 - r1
        period = 2 * gap
        for r in range(r1, H):
            offset = r - r1
            if offset % period == 0:
                for c in range(W): out[r][c] = v1
            elif offset % period == gap:
                for c in range(W): out[r][c] = v2
    return out


def _solve_fn_slide_to_adjacent(inp):
    """Non-8 shape slides toward 8-shape until bounding boxes are adjacent."""
    H, W = len(inp), len(inp[0])
    cells_8  = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 8]
    cells_nz = [(r, c) for r in range(H) for c in range(W)
                if inp[r][c] != 0 and inp[r][c] != 8]
    if not cells_8 or not cells_nz:
        return None
    r8_min = min(r for r, c in cells_8); r8_max = max(r for r, c in cells_8)
    c8_min = min(c for r, c in cells_8); c8_max = max(c for r, c in cells_8)
    rn_min = min(r for r, c in cells_nz); rn_max = max(r for r, c in cells_nz)
    cn_min = min(c for r, c in cells_nz); cn_max = max(c for r, c in cells_nz)
    dr = dc = 0
    if   rn_max < r8_min: dr = r8_min - rn_max - 1
    elif rn_min > r8_max: dr = r8_max - rn_min + 1
    elif cn_max < c8_min: dc = c8_min - cn_max - 1
    elif cn_min > c8_max: dc = c8_max - cn_min + 1
    else: return None
    out = [[0] * W for _ in range(H)]
    for r, c in cells_8:
        out[r][c] = 8
    for r, c in cells_nz:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr][nc] = inp[r][c]
    return out


def _solve_fn_extract_unique_region(inp):
    """Crop to the rectangular region containing the colour unique to that region."""
    H, W = len(inp), len(inp[0])
    zero_rows = set(r for r in range(H) if all(inp[r][c] == 0 for c in range(W)))
    zero_cols = set(c for c in range(W) if all(inp[r][c] == 0 for r in range(H)))

    def segments(zeros, total):
        segs, start = [], None
        for i in range(total):
            if i not in zeros:
                if start is None: start = i
            elif start is not None:
                segs.append((start, i)); start = None
        if start is not None: segs.append((start, total))
        return segs

    row_segs = segments(zero_rows, H)
    col_segs = segments(zero_cols, W)
    if not row_segs or not col_segs:
        return None

    from collections import Counter
    colour_count: Counter = Counter()
    region_colours: dict = {}
    for r0, r1 in row_segs:
        for c0, c1 in col_segs:
            colours = set(inp[r][c] for r in range(r0, r1) for c in range(c0, c1)
                          if inp[r][c] != 0)
            region_colours[(r0, r1, c0, c1)] = colours
            colour_count.update(colours)

    unique = {col for col, cnt in colour_count.items() if cnt == 1}
    if not unique:
        return None
    for (r0, r1, c0, c1), colours in region_colours.items():
        if colours & unique:
            return [list(inp[r][c0:c1]) for r in range(r0, r1)]
    return None


def _solve_fn_extend_period(inp):
    """Extend a periodic row pattern by H//2 rows, recolouring 1→2."""
    H, W = len(inp), len(inp[0])
    period = H
    for P in range(1, H):
        if all(inp[i] == inp[i % P] for i in range(H)):
            period = P
            break
    out_H = H * 3 // 2
    out = []
    for i in range(out_H):
        row = [2 if v == 1 else v for v in inp[i % period]]
        out.append(row)
    return out


# ── Agent-verified solvers (tasks 10fcaaa3 – 1caeab9d) ───────────────────────

def _solve_fn_diagonal_tile_2x2(inp):
    """Tile input 2×2; fill empty cells with colour 8 where diag-adjacent to non-zero."""
    H, W = len(inp), len(inp[0])
    tiled = [[inp[r % H][c % W] for c in range(2 * W)] for r in range(2 * H)]
    result = [row[:] for row in tiled]
    for r in range(2 * H):
        for c in range(2 * W):
            if tiled[r][c] == 0:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 2 * H and 0 <= nc < 2 * W and tiled[nr][nc] != 0:
                        result[r][c] = 8
                        break
    return result


def _solve_fn_complete_reflection(inp):
    """Complete 2D symmetry around the centre of the non-zero bounding box."""
    H, W = len(inp), len(inp[0])
    nz_rows = [r for r in range(H) if any(inp[r][c] != 0 for c in range(W))]
    nz_cols = [c for c in range(W) if any(inp[r][c] != 0 for r in range(H))]
    if not nz_rows or not nz_cols:
        return [row[:] for row in inp]
    row_axis = (nz_rows[0] + nz_rows[-1]) / 2.0
    col_axis = (nz_cols[0] + nz_cols[-1]) / 2.0
    result = [row[:] for row in inp]
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0:
                for nr in {r, int(round(2 * row_axis - r))}:
                    for nc in {c, int(round(2 * col_axis - c))}:
                        if 0 <= nr < H and 0 <= nc < W:
                            result[nr][nc] = v
    return result


def _solve_fn_sep_grid_dimensions(inp):
    """Separator-colour lines form a grid; output = (row_bands × col_bands) of bg colour."""
    from collections import Counter
    H, W = len(inp), len(inp[0])
    all_vals = [v for row in inp for v in row]
    sep_color = None
    for color in set(all_vals) - {0}:
        if any(all(inp[r][c] == color for c in range(W)) for r in range(H)):
            sep_color = color; break
        if any(all(inp[r][c] == color for r in range(H)) for c in range(W)):
            sep_color = color; break
    if sep_color is None:
        return None
    sep_rows = [r for r in range(H) if all(inp[r][c] == sep_color for c in range(W))]
    sep_cols = [c for c in range(W) if all(inp[r][c] == sep_color for r in range(H))]
    n_row_bands = len(sep_rows) + 1
    n_col_bands = len(sep_cols) + 1
    bg_counts = Counter(v for v in all_vals if v != 0 and v != sep_color)
    if not bg_counts:
        return None
    bg = bg_counts.most_common(1)[0][0]
    return [[bg] * n_col_bands for _ in range(n_row_bands)]


def _solve_fn_overlay_neighbourhood(inp):
    """Overlay 3×3 neighbourhood of every colour-5 cell onto a shared 3×3 output."""
    H, W = len(inp), len(inp[0])
    result = [[0] * 3 for _ in range(3)]
    for r in range(H):
        for c in range(W):
            if inp[r][c] == 5:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] != 0:
                            result[1 + dr][1 + dc] = inp[nr][nc]
    return result


def _solve_fn_tile_pack_grey(inp):
    """Pack grey (5) region with 2×2 (→8) and 3×1/1×3 (→2) tiles via backtracking."""
    H, W = len(inp), len(inp[0])
    target = frozenset((r, c) for r in range(H) for c in range(W) if inp[r][c] == 5)
    if not target:
        return [row[:] for row in inp]
    cells_sorted = sorted(target)

    def next_cell(filled):
        for pos in cells_sorted:
            if pos not in filled:
                return pos
        return None

    def backtrack(filled):
        pos = next_cell(filled)
        if pos is None:
            return filled
        r, c = pos
        for blocks, val in [
            ([(r, c), (r, c+1), (r+1, c), (r+1, c+1)], 8),  # 2×2
            ([(r, c), (r, c+1), (r, c+2)], 2),               # 1×3
            ([(r, c), (r+1, c), (r+2, c)], 2),               # 3×1
        ]:
            if all(
                0 <= br < H and 0 <= bc < W and (br, bc) in target and (br, bc) not in filled
                for br, bc in blocks
            ):
                nf = dict(filled)
                for br, bc in blocks:
                    nf[(br, bc)] = val
                res = backtrack(nf)
                if res is not None:
                    return res
        return None

    solution = backtrack({})
    result = [row[:] for row in inp]
    if solution:
        for (r, c), val in solution.items():
            result[r][c] = val
    return result


def _solve_fn_snap_to_separator(inp):
    """Move stray sep-colour cells to the adjacent position nearest their separator line."""
    H, W = len(inp), len(inp[0])
    sep_rows: dict[int, list] = {}
    sep_cols: dict[int, list] = {}
    for r in range(H):
        vals = set(inp[r][c] for c in range(W))
        if len(vals) == 1 and 0 not in vals:
            col = next(iter(vals))
            sep_rows.setdefault(col, []).append(r)
    for c in range(W):
        vals = set(inp[r][c] for r in range(H))
        if len(vals) == 1 and 0 not in vals:
            col = next(iter(vals))
            sep_cols.setdefault(col, []).append(c)

    result = [[0] * W for _ in range(H)]
    sep_row_set = set(r for v in sep_rows.values() for r in v)
    sep_col_set = set(c for v in sep_cols.values() for c in v)
    for color, rows in sep_rows.items():
        for r in rows:
            for c in range(W):
                result[r][c] = color
    for color, cols in sep_cols.items():
        for sc in cols:
            for r in range(H):
                result[r][sc] = color

    all_sep_colors = set(sep_rows) | set(sep_cols)
    for r in range(H):
        if r in sep_row_set:
            continue
        for c in range(W):
            if c in sep_col_set:
                continue
            val = inp[r][c]
            if val == 0 or val not in all_sep_colors:
                continue
            best_dist, best_pos = float('inf'), None
            for sr in sep_rows.get(val, []):
                d = abs(r - sr)
                if d < best_dist:
                    best_dist = d
                    best_pos = (sr - 1 if r < sr else sr + 1, c)
            for sc in sep_cols.get(val, []):
                d = abs(c - sc)
                if d < best_dist:
                    best_dist = d
                    best_pos = (r, sc - 1 if c < sc else sc + 1)
            if best_pos:
                nr, nc = best_pos
                if 0 <= nr < H and 0 <= nc < W:
                    result[nr][nc] = val
    return result


def _solve_fn_two_dot_frame(inp):
    """Two input dots define upper/lower half frame borders."""
    H, W = len(inp), len(inp[0])
    dots = sorted(
        [(inp[r][c], r) for r in range(H) for c in range(W) if inp[r][c] != 0],
        key=lambda x: x[1]
    )
    if len(dots) != 2:
        return None
    (c1, r1), (c2, r2) = dots[0], dots[1]
    mid = (r1 + r2) // 2
    result = [[0] * W for _ in range(H)]
    for r in range(H):
        if r <= mid:
            color = c1
            full = (r == 0 or r == r1)
        else:
            color = c2
            full = (r == r2 or r == H - 1)
        if full:
            for c in range(W):
                result[r][c] = color
        else:
            result[r][0] = color
            result[r][W - 1] = color
    return result


def _solve_fn_extract_interior(inp):
    """Output the cells strictly inside the rectangular border of a single colour."""
    from collections import defaultdict
    H, W = len(inp), len(inp[0])
    color_cells: dict = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if inp[r][c] != 0:
                color_cells[inp[r][c]].append((r, c))
    for color, cells in color_cells.items():
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for r in range(r0, r1 + 1):
            border.add((r, c0)); border.add((r, c1))
        for c in range(c0, c1 + 1):
            border.add((r0, c)); border.add((r1, c))
        if set(cells) == border:
            return [[inp[r][c] for c in range(c0 + 1, c1)] for r in range(r0 + 1, r1)]
    return None


def _solve_fn_stamp_rotated(inp):
    """Connected template shapes + isolated marker cells. Find the D4 rotation/reflection
    that aligns each template's special (non-skeleton) cells with a marker set, then place
    each template in that orientation. Everything else is cleared."""
    H, W = len(inp), len(inp[0])
    from collections import Counter
    cells = {(r, c): inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0}

    def _d4(r, c, t):
        return [(r,c),(-c,r),(-r,-c),(c,-r),(r,-c),(-c,-r),(-r,c),(c,r)][t]

    markers = {p: v for p, v in cells.items()
               if not any((p[0]+dr, p[1]+dc) in cells
                          for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])}

    template_pos = set(cells) - set(markers)
    visited = set(); templates = []
    for start in template_pos:
        if start in visited: continue
        comp = {}; stack = [start]
        while stack:
            p = stack.pop()
            if p in visited: continue
            visited.add(p); comp[p] = cells[p]
            r, c = p
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if (nr, nc) in template_pos and (nr, nc) not in visited:
                    stack.append((nr, nc))
        templates.append(comp)

    out = [[0] * W for _ in range(H)]
    for template in templates:
        skeleton = Counter(template.values()).most_common(1)[0][0]
        min_r = min(r for r,c in template); min_c = min(c for r,c in template)
        norm = {(r-min_r, c-min_c): v for (r,c),v in template.items()}
        placed = False
        for t in range(8):
            trans = {_d4(r,c,t): v for (r,c),v in norm.items()}
            mr2 = min(r for r,c in trans); mc2 = min(c for r,c in trans)
            trans = {(r-mr2, c-mc2): v for (r,c),v in trans.items()}
            specials = [(r,c,v) for (r,c),v in trans.items() if v != skeleton]
            if not specials: continue
            for ar, ac, av in specials:
                for (mpr, mpc), mv in markers.items():
                    if mv != av: continue
                    dr, dc = mpr-ar, mpc-ac
                    if all(markers.get((sr+dr,sc+dc))==sv for sr,sc,sv in specials):
                        for (r,c),v in trans.items():
                            nr,nc = r+dr,c+dc
                            if 0<=nr<H and 0<=nc<W:
                                out[nr][nc] = v
                        placed = True; break
                if placed: break
            if placed: break
    return out


def _solve_fn_rotation_complete(inp):
    """Shape is almost 180° rotationally symmetric. Extra cells (those without a rotated
    counterpart) are stamped at their 180°-rotated positions in colour 2."""
    H, W = len(inp), len(inp[0])
    from collections import Counter
    cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 1]
    if not cells:
        return None
    cell_set = set(cells)
    candidates = Counter()
    for r1, c1 in cells:
        for r2, c2 in cells:
            candidates[((r1 + r2) / 2, (c1 + c2) / 2)] += 1
    best_center, best_count = None, -1
    for (pr, pc) in candidates:
        matched = sum(1 for r, c in cells if (2*pr - r, 2*pc - c) in cell_set)
        if matched > best_count:
            best_count, best_center = matched, (pr, pc)
    if best_center is None:
        return None
    pr, pc = best_center
    out = [row[:] for row in inp]
    for r, c in cells:
        nr, nc = int(round(2*pr - r)), int(round(2*pc - c))
        if (nr, nc) not in cell_set and 0 <= nr < H and 0 <= nc < W:
            out[nr][nc] = 2
    return out


def _solve_fn_stamp_with_arms(inp):
    """Template shape + marker clusters of other colours.
    Each marker cluster is a partial copy of the template placed at the first copy position.
    Direction inferred from offset; copies repeat (gap=1) to grid edge."""
    H, W = len(inp), len(inp[0])
    from collections import defaultdict
    colour_cells = defaultdict(list)
    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0:
                colour_cells[v].append((r, c))
    if not colour_cells:
        return None
    template_colour = max(colour_cells, key=lambda v: len(colour_cells[v]))
    template_cells = set(colour_cells[template_colour])
    out = [row[:] for row in inp]

    def _gap_is_one(tc, dr, dc):
        copy = {(r+dr, c+dc) for r,c in tc}
        tr = [r for r,c in tc]; tcc = [c for r,c in tc]
        cr = [r for r,c in copy]; cc = [c for r,c in copy]
        if dr > 0 and min(cr)-max(tr)-1 != 1: return False
        if dr < 0 and min(tr)-max(cr)-1 != 1: return False
        if dc > 0 and min(cc)-max(tcc)-1 != 1: return False
        if dc < 0 and min(tcc)-max(cc)-1 != 1: return False
        return True

    def _components(cells):
        cells = set(cells); visited = set(); comps = []
        for start in sorted(cells):
            if start in visited:
                continue
            comp = []; stack = [start]
            while stack:
                cell = stack.pop()
                if cell in visited: continue
                visited.add(cell); comp.append(cell)
                r, c = cell
                for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if (nr,nc) in cells and (nr,nc) not in visited:
                        stack.append((nr,nc))
            comps.append(comp)
        return comps

    for colour, cells in colour_cells.items():
        if colour == template_colour:
            continue
        for cluster in _components(cells):
            mr0, mc0 = cluster[0]
            offset = None
            for sr, sc in template_cells:
                dr, dc = mr0-sr, mc0-sc
                if not (dr or dc): continue
                if not all((mr-dr,mc-dc) in template_cells for mr,mc in cluster): continue
                if frozenset((r+dr,c+dc) for r,c in template_cells) & template_cells: continue
                if _gap_is_one(template_cells, dr, dc):
                    offset = (dr, dc); break
            if offset is None:
                continue
            dr, dc = offset
            n = 1
            while True:
                stamp = [(r+n*dr,c+n*dc) for r,c in template_cells]
                in_bounds = [(r,c) for r,c in stamp if 0<=r<H and 0<=c<W]
                if not in_bounds: break
                for r,c in in_bounds:
                    out[r][c] = colour
                n += 1
    return out


def _solve_fn_parallelogram_correct(inp):
    """Each colour component: keep bottom row + rightmost cell of second-to-last row fixed;
    shift all other cells one column right."""
    H, W = len(inp), len(inp[0])
    colours = set(inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != 0)
    out = [[0] * W for _ in range(H)]
    for colour in colours:
        cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == colour]
        max_row = max(r for r, c in cells)
        fixed = set((r, c) for r, c in cells if r == max_row)
        above = [(r, c) for r, c in cells if r == max_row - 1]
        if above:
            fixed.add(max(above, key=lambda x: x[1]))
        for r, c in cells:
            if (r, c) in fixed:
                out[r][c] = colour
            else:
                if c + 1 >= W:
                    return None
                out[r][c + 1] = colour
    return out


def _solve_fn_align_to_anchor(inp):
    """Shift every non-anchor colour block so its top row aligns with colour-1's top row."""
    H, W = len(inp), len(inp[0])
    one_cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 1]
    if not one_cells:
        return None
    anchor_top = min(r for r, c in one_cells)
    result = [[0] * W for _ in range(H)]
    for r, c in one_cells:
        result[r][c] = 1
    other_colors = set(inp[r][c] for r in range(H) for c in range(W)) - {0, 1}
    for color in other_colors:
        cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == color]
        top = min(r for r, c in cells)
        shift = anchor_top - top
        for r, c in cells:
            nr = r + shift
            if 0 <= nr < H:
                result[nr][c] = color
    return result


# ── Wrappers for batch-1 solvers ──────────────────────────────────────────────

_solve_diagonal_tile        = _make_category_solver(_solve_fn_diagonal_tile)
_solve_repeating_stripes    = _make_category_solver(_solve_fn_repeating_stripes)
_solve_slide_to_adjacent    = _make_category_solver(_solve_fn_slide_to_adjacent)
_solve_extract_unique_region = _make_category_solver(_solve_fn_extract_unique_region)
_solve_extend_period        = _make_category_solver(_solve_fn_extend_period)
_solve_diagonal_tile_2x2   = _make_category_solver(_solve_fn_diagonal_tile_2x2)
_solve_complete_reflection  = _make_category_solver(_solve_fn_complete_reflection)
_solve_sep_grid_dimensions  = _make_category_solver(_solve_fn_sep_grid_dimensions)
_solve_overlay_neighbourhood = _make_category_solver(_solve_fn_overlay_neighbourhood)
_solve_tile_pack_grey       = _make_category_solver(_solve_fn_tile_pack_grey)
_solve_snap_to_separator    = _make_category_solver(_solve_fn_snap_to_separator)
_solve_two_dot_frame        = _make_category_solver(_solve_fn_two_dot_frame)
_solve_extract_interior     = _make_category_solver(_solve_fn_extract_interior)
_solve_align_to_anchor          = _make_category_solver(_solve_fn_align_to_anchor)
_solve_parallelogram_correct    = _make_category_solver(_solve_fn_parallelogram_correct)
_solve_stamp_with_arms          = _make_category_solver(_solve_fn_stamp_with_arms)
_solve_rotation_complete        = _make_category_solver(_solve_fn_rotation_complete)
_solve_stamp_rotated            = _make_category_solver(_solve_fn_stamp_rotated)


def _solve_fn_sep_grid_connect(inp):
    """Separator-grid row/column connect: fill cells between same-colour pairs.

    Detects a regular separator grid (single colour forming full rows and columns).
    For each pair of same-colour cells in the same cell-row or cell-column, fills
    all cells between them (inclusive) with that colour.
    """
    H, W = len(inp), len(inp[0])
    if H < 3 or W < 3:
        return None

    # Find separator colour: a colour forming at least one complete row AND column
    sep_colour = None
    for r in range(H):
        colours = set(inp[r])
        if len(colours) == 1 and list(colours)[0] != 0:
            cand = list(colours)[0]
            # Verify it also appears as full columns somewhere
            if any(all(inp[rr][c] == cand for rr in range(H)) for c in range(W)):
                sep_colour = cand
                break
    if sep_colour is None:
        return None

    sep_rows = set(r for r in range(H) if all(inp[r][c] == sep_colour for c in range(W)))
    sep_cols = set(c for c in range(W) if all(inp[r][c] == sep_colour for r in range(H)))
    if not sep_rows or not sep_cols:
        return None

    # Build cell row / col groups
    def _groups(n, seps):
        groups, start = [], None
        for i in range(n):
            if i in seps:
                if start is not None:
                    groups.append((start, i - 1))
                    start = None
            else:
                if start is None:
                    start = i
        if start is not None:
            groups.append((start, n - 1))
        return groups

    row_groups = _groups(H, sep_rows)
    col_groups = _groups(W, sep_cols)
    if len(row_groups) < 2 or len(col_groups) < 2:
        return None

    # Build cell colour grid
    def cell_colour(i, j):
        r0, r1 = row_groups[i]
        c0, c1 = col_groups[j]
        colours = {inp[r][c] for r in range(r0, r1+1)
                   for c in range(c0, c1+1)
                   if inp[r][c] != 0 and inp[r][c] != sep_colour}
        if len(colours) > 1:
            return None
        return colours.pop() if colours else 0

    nr, nc = len(row_groups), len(col_groups)
    cg = [[cell_colour(i, j) for j in range(nc)] for i in range(nr)]
    if any(cg[i][j] is None for i in range(nr) for j in range(nc)):
        return None

    # Fill between same-colour pairs in same row or column
    filled = [row[:] for row in cg]
    for i in range(nr):
        row_vals = cg[i]
        for colour in {v for v in row_vals if v}:
            positions = [j for j, v in enumerate(row_vals) if v == colour]
            if len(positions) >= 2:
                for j in range(min(positions), max(positions) + 1):
                    if filled[i][j] == 0:
                        filled[i][j] = colour

    for j in range(nc):
        col_vals = [cg[i][j] for i in range(nr)]
        for colour in {v for v in col_vals if v}:
            positions = [i for i, v in enumerate(col_vals) if v == colour]
            if len(positions) >= 2:
                for i in range(min(positions), max(positions) + 1):
                    if filled[i][j] == 0:
                        filled[i][j] = colour

    # No change → not this solver
    if filled == cg:
        return None

    # Reconstruct output grid
    out = [row[:] for row in inp]
    for i in range(nr):
        r0, r1 = row_groups[i]
        for j in range(nc):
            c0, c1 = col_groups[j]
            if filled[i][j] != cg[i][j]:
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        out[r][c] = filled[i][j]
    return out


_solve_sep_grid_connect = _make_category_solver(_solve_fn_sep_grid_connect)


def _solve_fn_gravity_down(inp):
    """Gravity down: each column's non-zero cells fall to the bottom, preserving order."""
    H, W = len(inp), len(inp[0])
    out = [[0] * W for _ in range(H)]
    for c in range(W):
        non_zero = [inp[r][c] for r in range(H) if inp[r][c] != 0]
        for k, v in enumerate(non_zero):
            out[H - len(non_zero) + k][c] = v
    return out


_solve_gravity_down = _make_category_solver(_solve_fn_gravity_down)


def _solve_fn_sep_grid_stamp_master(inp):
    """Separator-grid stamp-master: stamp densest cell's shape into every other cell.

    Finds the cell with the most non-zero content (the master). For every other
    cell, stamps the master's shape: positions that are empty get the separator
    colour; positions already filled keep their existing colour.
    """
    H, W = len(inp), len(inp[0])
    if H < 3 or W < 3:
        return None

    # Find separator colour
    sep_colour = None
    for r in range(H):
        cs = set(inp[r])
        if len(cs) == 1 and list(cs)[0] != 0:
            cand = list(cs)[0]
            if any(all(inp[rr][c] == cand for rr in range(H)) for c in range(W)):
                sep_colour = cand
                break
    if sep_colour is None:
        return None

    sep_rows = set(r for r in range(H) if all(inp[r][c] == sep_colour for c in range(W)))
    sep_cols = set(c for c in range(W) if all(inp[r][c] == sep_colour for r in range(H)))

    def build_groups(n, seps):
        groups, start = [], None
        for i in range(n):
            if i in seps:
                if start is not None:
                    groups.append((start, i - 1)); start = None
            else:
                if start is None:
                    start = i
        if start is not None:
            groups.append((start, n - 1))
        return groups

    rg = build_groups(H, sep_rows)
    cg = build_groups(W, sep_cols)
    nr, nc = len(rg), len(cg)
    if nr < 2 or nc < 2:
        return None

    cr = rg[0][1] - rg[0][0] + 1  # cell height
    cc = cg[0][1] - cg[0][0] + 1  # cell width
    if any(g[1] - g[0] + 1 != cr for g in rg) or any(g[1] - g[0] + 1 != cc for g in cg):
        return None  # non-uniform cells

    def cell_nonzero(i, j):
        r0, c0 = rg[i][0], cg[j][0]
        return {(dr, dc) for dr in range(cr) for dc in range(cc)
                if inp[r0 + dr][c0 + dc] != 0}

    # Find master (most non-zero content cells)
    master_pos, best_count, mi, mj = set(), 0, 0, 0
    for i in range(nr):
        for j in range(nc):
            pos = cell_nonzero(i, j)
            if len(pos) > best_count:
                best_count, master_pos, mi, mj = len(pos), pos, i, j
    if best_count == 0:
        return None

    # Stamp master pattern into every cell; preserve existing non-zero content
    changed = False
    out = [row[:] for row in inp]
    for i in range(nr):
        for j in range(nc):
            r0, c0 = rg[i][0], cg[j][0]
            for (dr, dc) in master_pos:
                if inp[r0 + dr][c0 + dc] == 0:
                    out[r0 + dr][c0 + dc] = sep_colour
                    changed = True
    if not changed:
        return None
    return out


_solve_sep_grid_stamp_master = _make_category_solver(_solve_fn_sep_grid_stamp_master)


def _solve_fn_diagonal_stamp_extend(inp):
    """2×2 bounding box with one main colour + colour-2 corner indicators.

    Each corner of the 2×2 box that holds colour 2 indicates a diagonal extension
    direction. The box is stamped repeatedly in each indicated direction (all cells
    become the main colour) until out of bounds.
    Corner → direction: TL(-1,-1), TR(-1,+1), BL(+1,-1), BR(+1,+1).
    """
    H, W = len(inp), len(inp[0])
    # Find all non-zero cells
    nz = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != 0]
    if not nz:
        return None
    colours = {v for _, _, v in nz}
    if 2 not in colours or len(colours) < 2:
        return None

    non2_cells = [(r, c) for r, c, v in nz if v != 2]
    two_cells  = [(r, c) for r, c, v in nz if v == 2]
    non2_colours = {inp[r][c] for r, c in non2_cells}
    if len(non2_colours) != 1:
        return None
    main_colour = next(iter(non2_colours))

    # Bounding box of all non-zero cells must be exactly 2×2
    all_cells = non2_cells + two_cells
    r_min = min(r for r, c in all_cells)
    r_max = max(r for r, c in all_cells)
    c_min = min(c for r, c in all_cells)
    c_max = max(c for r, c in all_cells)
    if r_max - r_min != 1 or c_max - c_min != 1:
        return None
    # All 4 corners must be covered by non-zero cells
    corners = {(r_min, c_min), (r_min, c_max), (r_max, c_min), (r_max, c_max)}
    if set(all_cells) != corners:
        return None

    # Directions indicated by corners that hold colour 2
    corner_dir = {
        (r_min, c_min): (-1, -1),
        (r_min, c_max): (-1, +1),
        (r_max, c_min): (+1, -1),
        (r_max, c_max): (+1, +1),
    }
    directions = [corner_dir[pos] for pos in two_cells if pos in corner_dir]
    if not directions:
        return None

    out = [[0] * W for _ in range(H)]
    # Stamp the 2×2 box (all cells → main colour) at each step in each direction
    def stamp(r0, c0):
        for dr in range(2):
            for dc in range(2):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r][c] = main_colour

    stamp(r_min, c_min)  # step 0
    for dr, dc in directions:
        step = 1
        while True:
            nr0, nc0 = r_min + dr * step, c_min + dc * step
            # Check at least one cell of the 2×2 box is in bounds
            if not any(0 <= nr0 + i < H and 0 <= nc0 + j < W for i in range(2) for j in range(2)):
                break
            stamp(nr0, nc0)
            step += 1

    return out


_solve_diagonal_stamp_extend = _make_category_solver(_solve_fn_diagonal_stamp_extend)


def _solve_fn_project_onto_rect(inp):
    """Isolated cells project onto the nearest face of the solid-8 rectangle.

    The rectangle is the connected solid block of colour 8. Each isolated non-8 cell
    not inside the rectangle projects onto the rectangle's border cell at the same
    row (left/right face) or column (top/bottom face). The border cell is recoloured
    to the isolated cell's colour. Whichever projection lands on a border cell is used;
    if both row and column projections are possible, prefer the shorter distance.
    """
    H, W = len(inp), len(inp[0])
    # Find the 8-rectangle bounding box
    eight_cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 8]
    if not eight_cells:
        return None
    r0 = min(r for r, c in eight_cells)
    r1 = max(r for r, c in eight_cells)
    c0 = min(c for r, c in eight_cells)
    c1 = max(c for r, c in eight_cells)
    # Verify it's solid
    if any(inp[r][c] != 8 for r in range(r0, r1+1) for c in range(c0, c1+1)):
        return None

    # Find isolated non-zero non-8 cells (outside the rectangle)
    isolated = [(r, c) for r in range(H) for c in range(W)
                if inp[r][c] != 0 and inp[r][c] != 8
                and not (r0 <= r <= r1 and c0 <= c <= c1)]
    if not isolated:
        return None

    out = [row[:] for row in inp]
    changed = False
    for r, c in isolated:
        colour = inp[r][c]
        # Column projection (above/below the rectangle)
        if c0 <= c <= c1:
            if r < r0:  # above → project onto top face
                out[r0][c] = colour; changed = True
            elif r > r1:  # below → project onto bottom face
                out[r1][c] = colour; changed = True
        # Row projection (left/right of the rectangle)
        if r0 <= r <= r1:
            if c < c0:  # left → project onto left face
                out[r][c0] = colour; changed = True
            elif c > c1:  # right → project onto right face
                out[r][c1] = colour; changed = True

    return out if changed else None


_solve_project_onto_rect = _make_category_solver(_solve_fn_project_onto_rect)


def _solve_fn_connect_diagonal_pairs(inp):
    """Connect same-colour cell pairs on the same diagonal.

    For each pair of same-colour cells where r-c is constant (main diagonal) or
    r+c is constant (anti-diagonal), fills all cells between them on that diagonal
    with the same colour.
    """
    H, W = len(inp), len(inp[0])
    from collections import defaultdict
    main_diag: dict = defaultdict(list)  # r-c → [(r,c)]
    anti_diag: dict = defaultdict(list)  # r+c → [(r,c)]
    colours: dict = {}

    for r in range(H):
        for c in range(W):
            v = inp[r][c]
            if v != 0:
                colours[(r, c)] = v
                main_diag[r - c].append((r, c))
                anti_diag[r + c].append((r, c))

    out = [row[:] for row in inp]
    changed = False

    for diag_pts, is_main in [(main_diag, True), (anti_diag, False)]:
        for key, pts in diag_pts.items():
            by_colour: dict = defaultdict(list)
            for p in pts:
                by_colour[colours[p]].append(p)
            for colour, cpts in by_colour.items():
                if len(cpts) < 2:
                    continue
                cpts_sorted = sorted(cpts)
                r_start, c_start = cpts_sorted[0]
                steps = cpts_sorted[-1][0] - r_start
                for k in range(steps + 1):
                    r = r_start + k
                    c = c_start + k if is_main else c_start - k
                    if inp[r][c] == 0:
                        out[r][c] = colour
                        changed = True

    return out if changed else None


_solve_connect_diagonal_pairs = _make_category_solver(_solve_fn_connect_diagonal_pairs)


def _solve_fn_nearest_border_fill(inp):
    """Colour-3 indicators replaced by the colour of the nearer of two opposite borders.

    Detects two opposite full-row (or full-column) borders of different non-zero,
    non-3 colours. Each interior cell of colour 3 is replaced by the border colour
    that is closer (measured in rows for row borders, columns for column borders).
    """
    H, W = len(inp), len(inp[0])
    # Try row borders
    full_rows = [(r, inp[r][0]) for r in range(H)
                 if all(inp[r][c] == inp[r][0] for c in range(W)) and inp[r][0] != 0]
    if len(full_rows) == 2:
        (r1, c1), (r2, c2) = full_rows[0], full_rows[1]
        if c1 != c2 and c1 != 3 and c2 != 3:
            out = [row[:] for row in inp]
            for r in range(H):
                for c in range(W):
                    if inp[r][c] == 3:
                        d1 = abs(r - r1)
                        d2 = abs(r - r2)
                        out[r][c] = c1 if d1 <= d2 else c2
            return out

    # Try column borders
    full_cols = [(c, inp[0][c]) for c in range(W)
                 if all(inp[r][c] == inp[0][c] for r in range(H)) and inp[0][c] != 0]
    if len(full_cols) == 2:
        (c1, v1), (c2, v2) = full_cols[0], full_cols[1]
        if v1 != v2 and v1 != 3 and v2 != 3:
            out = [row[:] for row in inp]
            for r in range(H):
                for c in range(W):
                    if inp[r][c] == 3:
                        d1 = abs(c - c1)
                        d2 = abs(c - c2)
                        out[r][c] = v1 if d1 <= d2 else v2
            return out
    return None


_solve_nearest_border_fill = _make_category_solver(_solve_fn_nearest_border_fill)


def _always(d): return True  # noqa: E731  — permissive pre-filter; verify() is the gate


# ── Primitive registry ────────────────────────────────────────────────────────
#
# Order matters: more specific solvers should come first.
# find_solver() returns the first one that passes verify().

ALL_PRIMITIVES = [
    ("COLOUR_BY_HEIGHT",              _applies_colour_by_height,     _solve_colour_by_height),
    ("FLOOD_FILL_ENCLOSED",           _applies_flood_fill_enclosed,  _solve_flood_fill_enclosed),
    ("EXPAND_CROSS",                  _applies_expand_cross,         _solve_expand_cross),
    ("COLOUR_HALO",                   _applies_colour_halo,          _solve_colour_halo),
    ("COLOUR_SUBSTITUTION",           _applies_colour_subst,         _solve_colour_subst),
    ("COLOUR_REMOVAL",                _applies_colour_removal,       _solve_colour_removal),
    ("UNIFORM_ROW_MARK",              _applies_uniform_row_mark,     _solve_uniform_row_mark),
    ("MIRROR_AT_MARKER",              _applies_mirror_at_marker,     _solve_mirror_at_marker),
    ("SHIFT_DOWN_ONE",                _applies_shift_down_one,       _solve_shift_down_one),
    ("CROP_BOUNDING_BOX",             _applies_crop_bounding_box,    _solve_crop_bounding_box),
    ("LOGICAL_OP",                    _always,                       _solve_logical_op),
    ("LOGICAL_AND",                   _applies_logical_and,          _solve_logical_and),
    # ── category-module solvers ──
    ("RECTANGLE_FROM_CORNERS",        _always, _solve_rectangle_from_corners),
    ("GAP_BRIDGE",                    _always, _solve_gap_bridge),
    ("SEPARATOR_GRID_CROSS_FILL",     _always, _solve_separator_grid_cross),
    ("BOUNDING_BOX_FILL",             _always, _solve_bounding_box_fill),
    ("HOLE_FILL_2X2",                 _always, _solve_hole_fill_2x2),
    ("COLOUR_MARKER_CROSS",           _always, _solve_colour_marker_cross),
    ("VERTICAL_COMB",                 _always, _solve_vertical_comb),
    ("SEPARATOR_GRID_DIAGONAL_FILL",  _always, _solve_separator_grid_diagonal),
    ("BORDER_ENCODED_SCALE",          _always, _solve_border_encoded_scale),
    ("QUADRANT_REFLECT",              _always, _solve_quadrant_reflect),
    ("SELF_TILE",                     _always, _solve_self_tile),
    ("LINE_FILL_BY_COLOUR",           _always, _solve_line_fill_by_colour),
    ("ROW_FILL_MEET_MIDDLE",          _always, _solve_row_fill_meet_middle),
    ("CONNECT_ALIGNED_PAIRS",         _always, _solve_connect_aligned_pairs),
    ("QUADRANT_MIRROR",               _always, _solve_quadrant_mirror),
    ("GEOMETRIC_TRANSFORM",           _always, _solve_geometric_transform),
    ("COLOUR_REMAP",                  _always, _solve_colour_remap),
    ("TILE_FILL",                     _always, _solve_tile_fill),
    # ── batch-1 task-specific solvers ──
    ("DIAGONAL_TILE",                 _always, _solve_diagonal_tile),
    ("REPEATING_STRIPES",             _always, _solve_repeating_stripes),
    ("SLIDE_TO_ADJACENT",             _always, _solve_slide_to_adjacent),
    ("EXTRACT_UNIQUE_REGION",         _always, _solve_extract_unique_region),
    ("EXTEND_PERIOD",                 _always, _solve_extend_period),
    ("DIAGONAL_TILE_2X2",             _always, _solve_diagonal_tile_2x2),
    ("COMPLETE_REFLECTION",           _always, _solve_complete_reflection),
    ("SEP_GRID_DIMENSIONS",           _always, _solve_sep_grid_dimensions),
    ("OVERLAY_NEIGHBOURHOOD",         _always, _solve_overlay_neighbourhood),
    ("TILE_PACK_GREY",                _always, _solve_tile_pack_grey),
    ("SNAP_TO_SEPARATOR",             _always, _solve_snap_to_separator),
    ("TWO_DOT_FRAME",                 _always, _solve_two_dot_frame),
    ("EXTRACT_INTERIOR",              _always, _solve_extract_interior),
    ("ALIGN_TO_ANCHOR",               _always, _solve_align_to_anchor),
    ("PARALLELOGRAM_CORRECT",         _always, _solve_parallelogram_correct),
    ("STAMP_WITH_ARMS",               _always, _solve_stamp_with_arms),
    ("ROTATION_COMPLETE",             _always, _solve_rotation_complete),
    ("STAMP_ROTATED",                 _always, _solve_stamp_rotated),
    ("SEP_GRID_CONNECT",              _always, _solve_sep_grid_connect),
    ("GRAVITY_DOWN",                  _always, _solve_gravity_down),
    ("SEP_GRID_STAMP_MASTER",         _always, _solve_sep_grid_stamp_master),
    ("DIAGONAL_STAMP_EXTEND",         _always, _solve_diagonal_stamp_extend),
    ("PROJECT_ONTO_RECT",             _always, _solve_project_onto_rect),
    ("CONNECT_DIAGONAL_PAIRS",        _always, _solve_connect_diagonal_pairs),
    ("NEAREST_BORDER_FILL",           _always, _solve_nearest_border_fill),
]


# ── Solver dispatch ───────────────────────────────────────────────────────────

def find_solver(task: dict, primitives=None):
    """Try each applicable primitive in order.

    Returns (solver_name, test_predictions) for the first primitive that:
      1. passes the applies_fn delta pre-filter, AND
      2. passes verify() against all training pairs.

    Returns (None, None) if no primitive succeeds.
    """
    if primitives is None:
        primitives = ALL_PRIMITIVES

    d = task_delta(task)

    for name, applies_fn, solve_fn in primitives:
        if not applies_fn(d):
            continue
        if verify(solve_fn, task):
            preds = solve_fn(task)
            return name, preds

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Delta-driven ARC solver — run on all training tasks or trace one."
    )
    ap.add_argument("--task", default=None, help="Trace a single task ID")
    args = ap.parse_args()

    task_ids = sorted(p.stem for p in TRAINING_DIR.glob("*.json"))

    if args.task:
        task = load_task(args.task)
        d = task_delta(task)
        print(f"Task  : {args.task}")
        print(f"Delta : zeros_gained={d['zeros_gained']}  zeros_lost={d['zeros_lost']}  "
              f"recoloured={d['recoloured']}  unchanged={d['unchanged']}")
        print(f"        new_colours={d['new_colours']}  lost_colours={d['lost_colours']}")
        name, preds = find_solver(task)
        print(f"Solver: {name or 'None'}")
        if preds:
            for i, p in enumerate(preds):
                print(f"  test[{i}] prediction shape: {p.shape}")
        return

    # ── Run on all tasks ──────────────────────────────────────────────────────
    solved_by: dict[str, list[str]] = defaultdict(list)
    unsolved: list[str] = []

    for tid in task_ids:
        task = load_task(tid)
        name, _ = find_solver(task)
        if name:
            solved_by[name].append(tid)
        else:
            unsolved.append(tid)

    print(f"\n{'Solver':<30} {'Solved':>6}  Tasks")
    print("-" * 70)
    for name, tids in sorted(solved_by.items(), key=lambda x: -len(x[1])):
        sample = "  " + "  ".join(tids[:4]) + ("..." if len(tids) > 4 else "")
        print(f"  {name:<28} {len(tids):>6}{sample}")

    n_solved = sum(len(v) for v in solved_by.values())
    print(f"\n  Solved   : {n_solved} / {len(task_ids)}")
    print(f"  Unsolved : {len(unsolved)} / {len(task_ids)}")


if __name__ == "__main__":
    main()
