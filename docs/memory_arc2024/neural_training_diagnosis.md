---
name: Neural training diagnosis — arc_exact=0.000
description: Root cause analysis for why all neural training runs produced 0 exact matches; fix confirmed
type: project
originSessionId: e487d252-10cb-4121-a4a5-aef8e3f8d060
---
## Problem
Both C8 and C7 neural training runs showed arc_exact=0.000 throughout. Confirmed NOT a code bug.

## Root causes

### 1. Mixed-mechanism clusters
C8 (11 tasks, called "compartment_fill") actually contained 6 different mechanisms:
- Colour-replacement / fill
- Pattern completion
- Logical ops
- Geometric transforms
- etc.

The model learned structural patterns (separator lines) common to many tasks but not any task-specific rule.
6d0160f0 reached 96.3% cell_acc even in this polluted cluster — showing the architecture works fine for individual tasks.

C7 had mixed output sizes → model couldn't converge on a single rule.

### 2. Checkpoint not selected by arc_exact
`scripts/train_transformer.py`: `best_val_loss = 0.0 if args.val_arc_task_ids else float("inf")`.
If `--val-arc-task-ids` is NOT set, `validate_on_arc()` never runs and checkpoints are saved by RE-ARC val loss.
C8 and C7 runs did NOT set `--val-arc-task-ids` → checkpoint at epoch=186 had best_val_loss=0.0457 (RE-ARC), arc_exact never measured during training.

## Fix for next run
1. Pick a **coherent single-mechanism cluster** (all tasks same rule)
2. **Always set `--val-arc-task-ids`** in train_colab.ipynb CONFIGS so arc_exact drives checkpoint selection
3. Prime candidate: find the cluster containing 6d0160f0 (96.3% cell_acc even when polluted → very learnable)

## Key data point
evaluate.py on C8 checkpoint (epoch=186, best_val_loss=0.0457):
- 11 tasks, 35 leave-one-out queries
- Mean cell_acc: 0.343
- 6d0160f0: 96.3% cell_acc (best — clearly a coherent, learnable task)
- 1190e5a7: 0.000 cell_acc (worst — completely wrong mechanism in this cluster)
- 0 exact matches total
