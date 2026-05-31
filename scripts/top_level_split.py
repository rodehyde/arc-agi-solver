import json, numpy as np
from pathlib import Path

TRAINING_DIR = Path('data/training')
task_ids = sorted(p.stem for p in TRAINING_DIR.glob('*.json'))

same_size = []
single_cell = []
different_size = []

for tid in task_ids:
    raw = json.loads((TRAINING_DIR / f'{tid}.json').read_text())
    pairs = raw['train']

    all_single = all(
        np.array(p['output']).shape == (1, 1) for p in pairs
    )
    all_same = all(
        np.array(p['input']).shape == np.array(p['output']).shape for p in pairs
    )

    if all_single:
        single_cell.append(tid)
    elif all_same:
        same_size.append(tid)
    else:
        different_size.append(tid)

print(f'Same size (input.shape == output.shape) : {len(same_size)}')
print(f'Single cell output (1x1)               : {len(single_cell)}')
print(f'Different size (everything else)        : {len(different_size)}')
print(f'Total                                   : {len(same_size) + len(single_cell) + len(different_size)}')
