"""
run_geometric_solver.py — Evaluate the geometric transform solver on all training tasks.

For each task where detect_transform() fires, verifies that the solver
correctly predicts every training pair output (sanity check), then
reports coverage and the transform distribution.

Usage:
    conda run -n arc-agi python scripts/run_geometric_solver.py
    conda run -n arc-agi python scripts/run_geometric_solver.py --split evaluation
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loader import load_all_tasks
from src.categories.geometric_transforms import detect_transform, ALL_TRANSFORMS

PROJECT_ROOT = Path(__file__).parent.parent


def verify_on_training_pairs(task: dict, fn) -> bool:
    """Double-check: does fn(input) == output for every training pair?"""
    for p in task["train"]:
        inp = np.array(p["input"],  dtype=np.int32)
        out = np.array(p["output"], dtype=np.int32)
        if not np.array_equal(fn(inp), out):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = load_all_tasks(split=args.split)
    n_total = len(tasks)

    solved = []
    verification_failures = []
    transform_counts = Counter()

    for task in tasks:
        result = detect_transform(task)
        if result is None:
            continue
        name, fn = result
        # Sanity-check: verify all training pairs (detect_transform already does this,
        # but belt-and-braces)
        if not verify_on_training_pairs(task, fn):
            verification_failures.append(task["task_id"])
            continue
        solved.append((task["task_id"], name))
        transform_counts[name] += 1

    print(f"Split:   {args.split}")
    print(f"Total:   {n_total}")
    print(f"Solved:  {len(solved)}  ({100*len(solved)/n_total:.1f}%)")
    if verification_failures:
        print(f"WARNING: {len(verification_failures)} tasks failed post-detect verification: "
              f"{verification_failures}")

    print(f"\nTransform distribution:")
    for name, n in transform_counts.most_common():
        print(f"  {name:25s}: {n}")

    if args.verbose:
        print(f"\nSolved task IDs:")
        for tid, name in sorted(solved):
            print(f"  {tid}  {name}")


if __name__ == "__main__":
    main()
