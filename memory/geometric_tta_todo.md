---
name: Geometric TTA — DONE
description: D4 geometric TTA implemented in evaluate.py; all_400_arc training queued
type: project
originSessionId: e487d252-10cb-4121-a4a5-aef8e3f8d060
---
## Status: IMPLEMENTED

D4 geometric TTA added to `scripts/evaluate.py` in session on 2026-05-12.

## What was implemented
- `_D4 = [(k, f) for f in (False, True) for k in range(4)]` — 8 orientations
- `_d4_apply(grid, k_rot, flip)` and `_d4_reverse(grid, k_rot, flip)` helpers
- `tta_decode` now loops over D4 orientations × colour permutations
- 90°/270° rotations correctly swap H↔W when passing to `greedy_decode`
- `n_d4` parameter threads through `tta_decode` → `ttt_decode` → `evaluate_task` → `main`
- `--n-d4` CLI flag (default 8; set to 1 for colour-only TTA)
- Total votes = n_d4 × n_perms (default 8×20=160)

## Note on non-geometric tasks
The original concern (D4 hurting non-geometric tasks) is real but was accepted as a trade-off for the all-400 model. The user can compare `--n-d4=1` vs `--n-d4=8` results.
