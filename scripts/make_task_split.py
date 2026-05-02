"""
make_task_split.py — Create a fixed 70/20/10 task-level split of all 400 ARC tasks.

Saves data/task_split.json:
  {
    "train": [...280 task IDs...],
    "val":   [...80 task IDs...],
    "eval":  [...40 task IDs...]
  }

Run once and commit the result.  The split is fixed by SEED and never changes.

Usage:
    python scripts/make_task_split.py
    python scripts/make_task_split.py --clusters C8_compartment_fill pattern_restoration C3_mirror
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
OUT_FILE     = PROJECT_ROOT / "data" / "task_split.json"

# Known cluster definitions (from train_colab.ipynb CONFIGS)
CLUSTER_TASKS = {
    "C8_compartment_fill": [
        "09629e4f", "1190e5a7", "1bfc4729", "1e32b0e9", "272f95fa",
        "29623171", "54d9e175", "6773b310", "6d0160f0", "7b6016b9", "941d9a10",
    ],
    "pattern_restoration": [
        "3345333e", "3631a71a", "9ecd008a", "b8825c91",
        "0dfd9992", "29ec7d0e", "484b58aa", "73251a56", "c3f564a4",
    ],
    "C3_mirror": [
        "3af2c5a8", "49d1d64f", "4c4377d9", "62c24649", "67e8384a",
        "6d0aefbc", "6fa7a44f", "7fe24cdd", "8be77c9e", "8d5021e8",
        "a416b8f3", "c9e6f938",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="*", default=list(CLUSTER_TASKS.keys()),
                    help="Cluster names to show per-cluster breakdown for")
    args = ap.parse_args()

    # Load all task IDs from data/training/
    all_ids = sorted(p.stem for p in TRAINING_DIR.glob("*.json"))
    print(f"Total tasks: {len(all_ids)}")

    # Fixed shuffle
    rng = random.Random(SEED)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    # Split
    n_train = int(len(shuffled) * 0.70)
    n_val   = int(len(shuffled) * 0.20)
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:n_train + n_val]
    eval_ids  = shuffled[n_train + n_val:]

    print(f"  train : {len(train_ids)}")
    print(f"  val   : {len(val_ids)}")
    print(f"  eval  : {len(eval_ids)}")

    # Per-cluster breakdown
    train_set = set(train_ids)
    val_set   = set(val_ids)
    eval_set  = set(eval_ids)

    print()
    for cluster_name in args.clusters:
        if cluster_name not in CLUSTER_TASKS:
            print(f"  {cluster_name}: unknown cluster")
            continue
        tasks = CLUSTER_TASKS[cluster_name]
        c_train = [t for t in tasks if t in train_set]
        c_val   = [t for t in tasks if t in val_set]
        c_eval  = [t for t in tasks if t in eval_set]
        print(f"  {cluster_name} ({len(tasks)} tasks):")
        print(f"    train ({len(c_train)}): {c_train}")
        print(f"    val   ({len(c_val)}):   {c_val}")
        print(f"    eval  ({len(c_eval)}):  {c_eval}")

    # Save
    split = {"train": train_ids, "val": val_ids, "eval": eval_ids}
    OUT_FILE.write_text(json.dumps(split, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
