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


# ── Primitive registry ────────────────────────────────────────────────────────
#
# Order matters: more specific solvers should come first.
# find_solver() returns the first one that passes verify().

ALL_PRIMITIVES = [
    ("COLOUR_BY_HEIGHT",     _applies_colour_by_height,     _solve_colour_by_height),
    ("FLOOD_FILL_ENCLOSED",  _applies_flood_fill_enclosed,  _solve_flood_fill_enclosed),
    # New primitives will be added here
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
