"""Analyse the 66 FILL_REGIONS tasks by their delta profiles."""
import json, numpy as np, sys
sys.path.insert(0, '.')
from pathlib import Path
from scripts.human_tree import load_task as ht_load, classify
from scripts.solvers import load_task, task_delta

TRAINING_DIR = Path('data/training')
task_ids = sorted(p.stem for p in TRAINING_DIR.glob('*.json'))

rows = []
for tid in task_ids:
    t = ht_load(tid)
    if classify(t) != 'FILL_REGIONS':
        continue
    st = load_task(tid)
    d = task_delta(st)
    n_pairs = len(st['train'])
    # Also check: how many new colours per pair on average?
    new_per_pair = len(d['new_colours'])
    rows.append((tid, d['zeros_gained'], d['zeros_lost'], d['recoloured'],
                 new_per_pair, n_pairs))

# Sort by zeros_gained
rows.sort(key=lambda x: -x[1])

print(f"{'Task':<12} {'gained':>7} {'lost':>5} {'recoloured':>10} {'new_cols':>8} {'pairs':>6}")
print('-' * 60)
for tid, gained, lost, recol, new_c, pairs in rows:
    print(f"  {tid:<10} {gained:>7} {lost:>5} {recol:>10} {new_c:>8} {pairs:>6}")
