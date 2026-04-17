"""
size_features.py — Categorise ARC tasks based on grid size and colour-count features.

Categories (a task may belong to multiple):
    SAME_SIZE              — input and output always have the same dimensions
    SAME_COLOUR_COUNT      — count of non-zero cells is equal in every train pair
    FIXED_OUTPUT           — all output grids are the same size
    FIXED_OUTPUT_VARY_IN   — output size is fixed but input sizes vary (stricter)
    SINGLE_CELL_OUTPUT     — output is always a 1×1 grid
    SHRINK                 — output is always smaller (by area) than the input
    GROW                   — output is always larger (by area) than the input
"""

from src.loader import grid_dims, count_nonzero, grid_area


CATEGORIES = [
    "SAME_SIZE",
    "SAME_COLOUR_COUNT",
    "FIXED_OUTPUT",
    "FIXED_OUTPUT_VARY_IN",
    "SINGLE_CELL_OUTPUT",
    "SHRINK",
    "GROW",
]


def categorise_task(task: dict) -> list[str]:
    """
    Return a list of category labels that apply to this task,
    based on its training examples only.
    """
    pairs = task["train"]
    categories = []

    # --- SAME_SIZE: every pair has input dims == output dims ---
    if all(grid_dims(p["input"]) == grid_dims(p["output"]) for p in pairs):
        categories.append("SAME_SIZE")

    # --- SAME_COLOUR_COUNT: non-zero cell count matches in every pair ---
    if all(count_nonzero(p["input"]) == count_nonzero(p["output"]) for p in pairs):
        categories.append("SAME_COLOUR_COUNT")

    # --- FIXED_OUTPUT: all output grids share the same dimensions ---
    output_dims = [grid_dims(p["output"]) for p in pairs]
    fixed_out = len(set(output_dims)) == 1
    if fixed_out:
        categories.append("FIXED_OUTPUT")

    # --- FIXED_OUTPUT_VARY_IN: output is fixed size AND inputs vary in size ---
    input_dims = [grid_dims(p["input"]) for p in pairs]
    if fixed_out and len(set(input_dims)) > 1:
        categories.append("FIXED_OUTPUT_VARY_IN")

    # --- SINGLE_CELL_OUTPUT: every output is exactly 1×1 ---
    if fixed_out and output_dims[0] == (1, 1):
        categories.append("SINGLE_CELL_OUTPUT")

    # --- SHRINK / GROW: output area vs input area ---
    input_areas  = [grid_area(p["input"])  for p in pairs]
    output_areas = [grid_area(p["output"]) for p in pairs]

    if all(o < i for i, o in zip(input_areas, output_areas)):
        categories.append("SHRINK")

    if all(o > i for i, o in zip(input_areas, output_areas)):
        categories.append("GROW")

    return categories
