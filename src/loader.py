"""
loader.py — ARC-AGI task loading utilities.

Each task is a JSON file with structure:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test":  [{"input": [[...]], "output": [[...]]}]
    }
"""

import json
import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def load_task(path: str | Path) -> dict:
    """Load a single ARC task JSON file. Attaches the task_id from the filename."""
    path = Path(path)
    with open(path) as f:
        task = json.load(f)
    task["task_id"] = path.stem
    return task


def load_all_tasks(split: str = "training") -> list[dict]:
    """
    Load all tasks for a given split ('training' or 'evaluation').
    Returns a list of task dicts, each with a 'task_id' key.
    """
    folder = DATA_DIR / split
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    tasks = []
    for json_file in sorted(folder.glob("*.json")):
        tasks.append(load_task(json_file))
    return tasks


def grid_dims(grid: list[list[int]]) -> tuple[int, int]:
    """Return (rows, cols) for a grid."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return rows, cols


def count_nonzero(grid: list[list[int]]) -> int:
    """Count cells with a value greater than 0 (coloured squares)."""
    return sum(cell != 0 for row in grid for cell in row)


def grid_area(grid: list[list[int]]) -> int:
    """Return total number of cells in a grid (rows * cols)."""
    rows, cols = grid_dims(grid)
    return rows * cols
