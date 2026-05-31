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
        say("output_is_multiple=YES  →  TILE_ASSEMBLY")
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

        if output_grows_in_free_dim(task):
            say("output_grows_in_free_dim=YES  →  ONE_DIM_EXTEND")
            return "ONE_DIM_EXTEND"

        say("output_shrinks_in_free_dim  →  ONE_DIM_CROP")
        return "ONE_DIM_CROP"

    # ── Grids different size (catch-all) ──────────────────────────────────────
    say("different_size")

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
    "MOVE_TO_STATIC", "MOVE_PART",
    "COLOUR_REMOVAL", "COLOUR_SUBSTITUTION",
    "TILE_ASSEMBLY",
    "GRID_SELECT_ELEMENT", "AND_HALVES",
    "ONE_DIM_EXTEND", "ONE_DIM_CROP",
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
