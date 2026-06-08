---
name: TTT implementation
description: TTT added to evaluate.py; layer freezing added; LOO-based TTT confirmed broken; TTT-for-test is the next step
type: project
originSessionId: e487d252-10cb-4121-a4a5-aef8e3f8d060
---
## Current state (2026-05-18)

### What's implemented in scripts/evaluate.py
- `ttt_fine_tune()`: copies model, optionally freezes first N blocks, trains with batch augmentation
  - `freeze_layers=-1` (default): freeze first 75% of blocks automatically (4 of 6 for all_400_arc)
  - `freeze_layers=0`: train all layers (original behaviour)
  - Cosine LR decay, save-best inner LOO, early stopping with patience
- `ttt_decode()`: calls ttt_fine_tune then TTA predict
- `evaluate_task(mode="ttt")`: LOO evaluation using ttt_decode per held-out pair
- `_run_selective_ttt()`: Phase 1 TTA (cached), Phase 2 TTT on non-perfect TTA tasks
- TTA cache: saves to results/tta_cache_*.json, reloaded on next run (skips Phase 1)

### CLI flags
- `--ttt-steps` (default 100)
- `--ttt-lr` (default 1e-4)
- `--ttt-eval-every` (default 5)
- `--ttt-patience` (default 5)
- `--ttt-freeze-layers` (default -1 = auto/75%)
- `--ttt-batch-size` (default 8)

### What failed and why
- LOO-based TTT hurts ALL tasks (0/14 helped in pilot, both with and without layer freezing)
- Inner LOO reaching 1.0000 is a WARNING SIGN (overfitting) not success
- Even 2 unfrozen layers + head is enough to memorize 3-4 examples in 40-75 steps
- Fundamental problem: LOO-based TTT ≠ how competition winners use TTT

### Root cause (diagnosed 2026-05-18)
We use TTT in the WRONG way:
- Our LOO-based TTT: fine-tune on N-1 training pairs → predict held-out Nth pair
- Competition winners' TTT: fine-tune on ALL N training pairs → predict actual TEST pair

Example: 007bbfb7 — TTA LOO = 0.998 (almost perfect), but [test] neural = 0.728.
The test input is harder than training inputs. LOO-based TTT cannot fix this.

## Next step: TTT-for-test (NOT YET IMPLEMENTED)

Add `ttt_for_test: bool = False` parameter to `evaluate_task`.

In the compare_rule_based section, change neural test prediction:
```python
if ttt_for_test and ttt_steps > 0 and len(train) >= 2:
    n_freeze = int(len(model.blocks) * 0.75) if ttt_freeze_layers < 0 else ttt_freeze_layers
    ttt_model = ttt_fine_tune(model, tok, train, ttt_steps, ttt_lr, device, rng,
                              k_ctx, ttt_batch_size, ttt_eval_every, ttt_patience,
                              verbose=False, freeze_layers=n_freeze)
    neural_pred = tta_decode(ttt_model, tok, ctx_all, test_in, H, W,
                             n_perms, device, rng, n_d4)
elif n_perms > 1:
    neural_pred = tta_decode(model, tok, ctx_all, test_in, H, W,
                             n_perms, device, rng, n_d4)
else:
    neural_pred = greedy_decode(model, tok, ctx_all, test_in, H, W, device)
```

In `_run_selective_ttt` Phase 2, pass `ttt_for_test=True` so non-perfect TTA tasks
get TTT applied to their actual test pair prediction.

Also fix: test pair currently uses greedy even when n_perms > 1 (should use TTA).
