"""
explore.py — Load ARC training tasks, categorise them, and print a summary report.

Usage:
    python src/explore.py
    python src/explore.py --split evaluation
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.loader import load_all_tasks
from src.categories import categorise_task, CATEGORIES


RESULTS_DIR = Path(__file__).parent.parent / "results"


def main(split: str = "training") -> None:
    print(f"\nLoading '{split}' tasks...")
    tasks = load_all_tasks(split)
    total = len(tasks)
    print(f"Total tasks loaded: {total}\n")

    # Categorise every task
    task_categories: dict[str, list[str]] = {}
    category_counts: dict[str, int] = defaultdict(int)

    for task in tasks:
        cats = categorise_task(task)
        task_categories[task["task_id"]] = cats
        for cat in cats:
            category_counts[cat] += 1

    uncategorised = sum(1 for cats in task_categories.values() if not cats)

    # Print report
    col_w = max(len(c) for c in CATEGORIES) + 2
    print(f"{'Category':<{col_w}} {'Count':>6}   {'% of tasks':>10}")
    print("-" * (col_w + 22))
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        pct = 100 * count / total if total else 0
        print(f"{cat:<{col_w}} {count:>6}   {pct:>9.1f}%")
    print("-" * (col_w + 22))
    print(f"{'Uncategorised':<{col_w}} {uncategorised:>6}   {100 * uncategorised / total:>9.1f}%\n")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"categories_{split}.json"
    with open(output_path, "w") as f:
        json.dump(task_categories, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorise ARC tasks by grid features.")
    parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    args = parser.parse_args()
    main(args.split)
