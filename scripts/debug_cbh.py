import sys
sys.path.insert(0, '.')
from scripts.solvers import load_task, task_delta, verify, _solve_colour_by_height
from scripts.human_tree import load_task as ht_load, classify
from pathlib import Path

TRAINING_DIR = Path('data/training')
task_ids = sorted(p.stem for p in TRAINING_DIR.glob('*.json'))

for tid in task_ids:
    t = ht_load(tid)
    if classify(t) != 'COLOUR_BY_HEIGHT':
        continue
    st = load_task(tid)
    d = task_delta(st)
    passes = verify(_solve_colour_by_height, st)
    print(f"{tid}: gained={d['zeros_gained']} lost={d['zeros_lost']} "
          f"recoloured={d['recoloured']} new={d['new_colours']}  verify={passes}")
