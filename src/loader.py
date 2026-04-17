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


def load_re_arc_task(path: str | Path, n_train: int = 3) -> dict:
    """
    Load a RE-ARC task file and return it in standard ARC format.

    RE-ARC files contain a flat list of 1,000 {input, output} examples.
    This selects the first n_train as the train set and the next one as test.

    Args:
        path:    Path to a RE-ARC task JSON file.
        n_train: Number of examples to use as training pairs (default 3).
    """
    path = Path(path)
    with open(path) as f:
        examples = json.load(f)
    return {
        "task_id": path.stem,
        "train": examples[:n_train],
        "test": examples[n_train: n_train + 1],
    }


def load_all_re_arc_tasks(n_train: int = 3) -> list[dict]:
    """
    Load all RE-ARC tasks from data/re_arc/ in standard ARC format.

    Raises FileNotFoundError if the re_arc folder doesn't exist —
    run scripts/download_re_arc.py first.
    """
    folder = DATA_DIR / "re_arc"
    if not folder.exists():
        raise FileNotFoundError(
            f"RE-ARC data not found at {folder}. "
            "Run: python scripts/download_re_arc.py"
        )
    task_files = [f for f in sorted(folder.glob("*.json"))
                  if not f.name.startswith("._") and f.stem != "metadata"]
    return [load_re_arc_task(f, n_train) for f in task_files]


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
