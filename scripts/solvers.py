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
