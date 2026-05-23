"""
make_task_split.py — Create a fixed 70/20/10 task-level split of all 400 ARC tasks.

Uses category-aware stratified sampling so every task category is proportionally
represented across train / val / eval.  Each task is assigned a 'primary category'
— the rarest category it belongs to — and tasks are split proportionally within
each primary-category group.  This guarantees that rare task types appear in all
three splits, not concentrated by chance in just one.

Saves data/task_split.json:
  {
    "train": [...280 task IDs...],
    "val":   [...80 task IDs...],
    "eval":  [...40 task IDs...]
  }

Run once and commit the result.  The split is fixed by SEED and never changes.

Usage:
    python scripts/make_task_split.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42

PROJECT_ROOT    = Path(__file__).parent.parent
TRAINING_DIR    = PROJECT_ROOT / "data" / "training"
CATEGORIES_FILE = PROJECT_ROOT / "results" / "categories_training.json"
OUT_FILE        = PROJECT_ROOT / "data" / "task_split.json"

RATIOS = (0.70, 0.20, 0.10)   # train / val / eval


def primary_category(task_id: str, task_cats: dict, cat_counts: dict) -> str:
    """Return the rarest category the task belongs to.

    Ties broken alphabetically so the assignment is deterministic.
    Tasks with no category get a synthetic '__none__' primary.
    """
    cats = task_cats.get(task_id, [])
    if not cats:
        return "__none__"
    return min(cats, key=lambda c: (cat_counts[c], c))


def split_group(tids: list[str], rng: random.Random) -> tuple[list, list, list]:
    """Split a group of task IDs into (train, val, eval) proportionally.

    Guarantees:
      • Groups with ≥ 2 tasks: at least 1 in val
      • Groups with ≥ 5 tasks: at least 1 in eval (10 % of 5 rounds to 1)
      • Single-task groups: task goes to train
    """
    items = list(tids)
    rng.shuffle(items)
    n = len(items)

    n_val  = max(1, round(n * RATIOS[1])) if n >= 2 else 0
    n_eval = max(1, round(n * RATIOS[2])) if n >= 5 else (round(n * RATIOS[2]) if n >= 3 else 0)

    # Cap so we never over-allocate
    n_val  = min(n_val,  n)
    n_eval = min(n_eval, n - n_val)
    n_train = n - n_val - n_eval

    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    all_ids = sorted(p.stem for p in TRAINING_DIR.glob("*.json"))
    print(f"Total tasks: {len(all_ids)}")

    if not CATEGORIES_FILE.exists():
        raise FileNotFoundError(
            f"{CATEGORIES_FILE} not found.\n"
            "Run:  python -m src.explore  to generate it first."
        )
    task_cats: dict[str, list[str]] = json.loads(CATEGORIES_FILE.read_text())

    # Count tasks per category
    cat_counts: dict[str, int] = defaultdict(int)
    for cats in task_cats.values():
        for c in cats:
            cat_counts[c] += 1

    # ── Assign primary category and group ────────────────────────────────────
    groups: dict[str, list[str]] = defaultdict(list)
    for tid in all_ids:
        pc = primary_category(tid, task_cats, cat_counts)
        groups[pc].append(tid)

    print(f"\nPrimary-category groups ({len(groups)}):")
    for pc, tids in sorted(groups.items(), key=lambda x: len(x[1])):
        print(f"  {pc:<32}  {len(tids):>3} tasks")

    # ── Split each group proportionally ──────────────────────────────────────
    rng = random.Random(SEED)
    train_ids, val_ids, eval_ids = [], [], []

    # Process smallest groups first — they are most at risk of being missed
    for pc, tids in sorted(groups.items(), key=lambda x: (len(x[1]), x[0])):
        tr, v, e = split_group(tids, rng)
        train_ids.extend(tr)
        val_ids.extend(v)
        eval_ids.extend(e)

    # ── Report totals ────────────────────────────────────────────────────────
    print(f"\nSplit totals:")
    print(f"  train : {len(train_ids)}  (target {round(len(all_ids) * RATIOS[0])})")
    print(f"  val   : {len(val_ids)}   (target {round(len(all_ids) * RATIOS[1])})")
    print(f"  eval  : {len(eval_ids)}  (target {round(len(all_ids) * RATIOS[2])})")

    # ── Per-category breakdown ────────────────────────────────────────────────
    train_set = set(train_ids)
    val_set   = set(val_ids)
    eval_set  = set(eval_ids)

    print(f"\n{'Category':<32} {'Total':>6} {'Train':>6} {'Val':>5} {'Eval':>5}")
    print("-" * 60)
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1]):
        tids = [t for t in all_ids if cat in task_cats.get(t, [])]
        n_tr = sum(1 for t in tids if t in train_set)
        n_v  = sum(1 for t in tids if t in val_set)
        n_e  = sum(1 for t in tids if t in eval_set)
        warn = "  ← no val" if n_v == 0 else ("  ← no eval" if n_e == 0 else "")
        print(f"  {cat:<30} {count:>5} {n_tr:>6} {n_v:>5} {n_e:>5}{warn}")

    # ── Save ─────────────────────────────────────────────────────────────────
    split = {
        "train": sorted(train_ids),
        "val":   sorted(val_ids),
        "eval":  sorted(eval_ids),
    }
    OUT_FILE.write_text(json.dumps(split, indent=2))
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
