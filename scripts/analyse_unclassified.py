"""Analyse the SAME_SIZE_UNCLASSIFIED and ONE_DIM_SAME_UNCLASSIFIED buckets."""
import json
import numpy as np
import sys
sys.path.insert(0, '.')
from pathlib import Path
from scripts.human_tree import load_task, classify

TRAINING_DIR = Path('data/training')
task_ids = sorted(p.stem for p in TRAINING_DIR.glob('*.json'))

same_size_uncl = []
one_dim_uncl = []
for tid in task_ids:
    task = load_task(tid)
    cat = classify(task)
    if cat == 'SAME_SIZE_UNCLASSIFIED':
        same_size_uncl.append(tid)
    elif cat == 'ONE_DIM_SAME_UNCLASSIFIED':
        one_dim_uncl.append(tid)

print(f'SAME_SIZE_UNCLASSIFIED: {len(same_size_uncl)}')
for tid in same_size_uncl:
    task = load_task(tid)
    details = []
    for p in task['train']:
        ic = set(int(x) for x in np.unique(p['input'])) - {0}
        oc = set(int(x) for x in np.unique(p['output'])) - {0}
        lost = sorted(ic - oc)
        new_c = sorted(oc - ic)
        details.append(f'lost={lost} new={new_c}')
    print(f'  {tid}: {details[0]}  (pairs={len(task["train"])})')

print()
print(f'ONE_DIM_SAME_UNCLASSIFIED: {len(one_dim_uncl)}')
for tid in one_dim_uncl:
    task = load_task(tid)
    for p in task['train']:
        hi, wi = p['input'].shape
        ho, wo = p['output'].shape
        same_h = hi == ho
        same_w = wi == wo
        ic = set(int(x) for x in np.unique(p['input'])) - {0}
        oc = set(int(x) for x in np.unique(p['output'])) - {0}
        print(f'  {tid}: in={p["input"].shape} out={p["output"].shape} '
              f'same_h={same_h} same_w={same_w} ic={sorted(ic)} oc={sorted(oc)}')
        break
