"""
evaluate.py — Inference (greedy / TTA / TTT) on a trained ARC checkpoint.

Three inference modes:
  greedy — single forward pass, argmax at each cell position
  tta    — N colour permutations, un-permute predictions, majority-vote per cell
  ttt    — fine-tune on task training pairs for M steps (with augmentation),
            save the best intermediate model (by LOO cell acc), then run TTA

Evaluation is leave-one-out over the original ARC training pairs for each task:
  for each pair i in task["train"]:
      context = all other training pairs
      query   = pair i (model predicts output from input)
      score   = compare prediction to pair i output

Usage (download checkpoint from Colab first):
    python scripts/evaluate.py --checkpoint checkpoints/transformer_cC8_compartment_fill_best.pt
    python scripts/evaluate.py --checkpoint ... --mode tta   --n-perms 20
    python scripts/evaluate.py --checkpoint ... --mode ttt   --ttt-steps 200 --n-perms 20
    python scripts/evaluate.py --checkpoint ... --mode all           # compare all three
    python scripts/evaluate.py --checkpoint ... --verbose            # per-pair breakdown
"""

import argparse
import copy
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.arc_tokenizer import (
    ArcTokenizer, VOCAB_SIZE,
    START_OUT, END,
    F_TOKEN, F_GRID,
)
from src.transformer_model import ArcTransformer

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "training"


# ---------------------------------------------------------------------------
# Rule-based solver dispatcher
# ---------------------------------------------------------------------------

def _load_rule_solvers():
    """Import rule-based solvers lazily (so evaluate.py works without them)."""
    try:
        from src.solvers.geometric   import solve_geometric
        from src.solvers.flood_fill  import solve_flood_fill
        from src.solvers.logical_ops import solve_logical_op
        from src.solvers.tiling      import solve_tile_fill, solve_tile_compress
        return [solve_geometric, solve_flood_fill, solve_logical_op,
                solve_tile_fill, solve_tile_compress]
    except ImportError as e:
        print(f"WARNING: rule-based solvers not available ({e}) — --compare-rule-based skipped")
        return []

_RULE_SOLVERS = None   # loaded on first use


def try_rule_based(task: dict) -> np.ndarray | None:
    """Try each rule-based solver in turn; return first non-None prediction.

    Solvers detect their own applicability from the training pairs.
    Returns a uint8 numpy array (test output prediction) or None if no solver fires.
    """
    global _RULE_SOLVERS
    if _RULE_SOLVERS is None:
        _RULE_SOLVERS = _load_rule_solvers()

    for solver in _RULE_SOLVERS:
        try:
            pred = solver(task)
            if pred is not None:
                return np.array(pred, dtype=np.uint8)
        except Exception:
            pass   # solver raised — treat as non-applicable
    return None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load checkpoint, reconstruct model, return (model, saved_args, task_ids)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved    = ckpt.get("args", {})
    task_ids = ckpt.get("task_ids", [])

    model = ArcTransformer(
        vocab_size   = VOCAB_SIZE,
        d_model      = saved.get("d_model",      256),
        n_heads      = saved.get("n_heads",      8),
        n_layers     = saved.get("n_layers",     6),
        max_seq_len  = saved.get("max_seq_len",  6000),
        max_grid_dim = saved.get("max_grid_dim", 32),
        dropout      = 0.0,   # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    best  = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded checkpoint  epoch={epoch}  best_val_loss={best:.4f}")
    print(f"Checkpoint task IDs: {task_ids}")
    return model, saved, task_ids


# ---------------------------------------------------------------------------
# Prefix construction for autoregressive generation
# ---------------------------------------------------------------------------

def build_prefix(
    tok:  ArcTokenizer,
    ctx:  list[tuple[np.ndarray, np.ndarray]],
    test_in: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build (prefix, pad_mask, test_out_grid_num) for model.generate().

    encode_sequence(..., test_output=None) produces:
        START  ctx_pairs...  START_IN test_in END_IN  END

    We drop the trailing END and append START_OUT with the test-output grid
    number, leaving the model to generate the grid token-by-token.
    """
    feats, _ = tok.encode_sequence(ctx, test_in, test_output=None)
    # feats[-1] is always END when test_output=None
    k = len(ctx)
    test_out_gnum = 2 * k + 2
    feats_trim = feats[:-1]   # drop END
    start_out  = np.array([[START_OUT, 0, 0, 0, test_out_gnum]], dtype=np.int16)
    feats_full = np.concatenate([feats_trim, start_out], axis=0)
    prefix   = torch.from_numpy(feats_full).unsqueeze(0).long()    # (1, T, 5)
    pad_mask = torch.zeros(1, prefix.shape[1], dtype=torch.bool)
    return prefix, pad_mask, test_out_gnum


# ---------------------------------------------------------------------------
# Error analysis helpers
# ---------------------------------------------------------------------------

def _severity(cell_acc: float) -> str:
    """Bucket a cell accuracy into a human-readable severity label."""
    if cell_acc >= 1.0: return "perfect"
    if cell_acc >= 0.9: return "close"
    if cell_acc >= 0.5: return "partial"
    if cell_acc > 0.0:  return "far"
    return "total"


def save_analysis(
    results:    list[dict],
    mode:       str,
    ckpt_path:  str,
    out_path:   Path,
) -> None:
    """Print a ranked error-analysis report and save full detail to JSON.

    The report covers:
    - Severity distribution (perfect / close / partial / far / total)
    - Bottom-N tasks by cell accuracy (worst failures)
    - Most inconsistent tasks (high variance across LOO pairs — best TTT candidates)
    - Most consistent failures (low variance, always wrong — systematic errors)
    """
    SEVERITY_ORDER = ["perfect", "close", "partial", "far", "total"]

    # ── Severity counts ───────────────────────────────────────────────────────
    sev_counts: dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
    for r in results:
        sev_counts[r["severity"]] += 1
    n = len(results)

    print(f"\n{'=' * 60}")
    print(f" ERROR ANALYSIS: {mode.upper()}")
    print(f"{'=' * 60}")
    print(f"\n{'Severity':<10} {'Cell-acc range':<18} {'Tasks':>6}  {'%':>5}")
    print("-" * 45)
    ranges = {
        "perfect": "100%",
        "close":   "90–99%",
        "partial": "50–89%",
        "far":     "1–49%",
        "total":   "0%",
    }
    for s in SEVERITY_ORDER:
        c = sev_counts[s]
        print(f"  {s:<8} {ranges[s]:<18} {c:>6}  {100*c/n:>4.0f}%")

    # ── Bottom 20 by cell accuracy ────────────────────────────────────────────
    bottom = sorted(results, key=lambda r: r["cell_acc"])[:20]
    print(f"\nBottom 20 tasks (worst cell accuracy):")
    print(f"  {'Task ID':<16} {'CellAcc':>7}  {'Exact':>7}  {'Sev':<8}  {'Output':>8}  {'AccStd':>6}")
    print(f"  {'-'*16} {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}")
    for r in bottom:
        out_size = f"{r.get('output_h','?')}×{r.get('output_w','?')}"
        print(f"  {r['task_id']:<16} {r['cell_acc']:>7.3f}  "
              f"{r['n_exact']:>2}/{r['n_pairs']:<4}  {r['severity']:<8}  "
              f"{out_size:>8}  {r.get('cell_acc_std', 0.0):>6.3f}")

    # ── Most inconsistent (high std, < perfect) — best TTT candidates ─────────
    improvable = [r for r in results if r["cell_acc"] < 1.0]
    inconsistent = sorted(improvable,
                          key=lambda r: r.get("cell_acc_std", 0.0),
                          reverse=True)[:10]
    print(f"\nTop 10 most inconsistent tasks (sometimes right — good TTT candidates):")
    print(f"  {'Task ID':<16} {'CellAcc':>7}  {'AccStd':>6}  {'Sev':<8}")
    print(f"  {'-'*16} {'-'*7}  {'-'*6}  {'-'*8}")
    for r in inconsistent:
        print(f"  {r['task_id']:<16} {r['cell_acc']:>7.3f}  "
              f"{r.get('cell_acc_std', 0.0):>6.3f}  {r['severity']:<8}")

    # ── Most consistent failures (low std, always wrong — systematic errors) ───
    systematic = [r for r in improvable
                  if r.get("cell_acc_std", 1.0) < 0.05 and r["cell_acc"] < 0.5]
    systematic = sorted(systematic, key=lambda r: r["cell_acc"])[:10]
    if systematic:
        print(f"\nTop 10 most consistent failures (always wrong — systematic errors):")
        print(f"  {'Task ID':<16} {'CellAcc':>7}  {'AccStd':>6}  {'AvgWrong':>8}")
        print(f"  {'-'*16} {'-'*7}  {'-'*6}  {'-'*8}")
        for r in systematic:
            print(f"  {r['task_id']:<16} {r['cell_acc']:>7.3f}  "
                  f"{r.get('cell_acc_std', 0.0):>6.3f}  "
                  f"{r.get('mean_n_wrong', 0.0):>8.1f}")

    # ── Rule-based vs neural comparison (test pairs) ─────────────────────────
    rb_results = [r for r in results if "rule_cmp" in r]
    if rb_results:
        fired      = [r for r in rb_results if r["rule_cmp"]["rule_fires"]]
        n_fired    = len(fired)
        n_rb_exact = sum(1 for r in fired if r["rule_cmp"]["rule_test_exact"])
        n_nn_exact = sum(1 for r in fired if r["rule_cmp"]["neural_test_exact"])
        n_agree    = sum(1 for r in fired if r["rule_cmp"]["agree"])
        n_rb_only  = sum(1 for r in fired
                         if r["rule_cmp"]["rule_test_exact"]
                         and not r["rule_cmp"]["neural_test_exact"])
        n_nn_only  = sum(1 for r in fired
                         if r["rule_cmp"]["neural_test_exact"]
                         and not r["rule_cmp"]["rule_test_exact"])
        n_both     = sum(1 for r in fired
                         if r["rule_cmp"]["rule_test_exact"]
                         and r["rule_cmp"]["neural_test_exact"])

        print(f"\nRule-based vs neural (on actual test pairs):")
        print(f"  Rule-based fired on {n_fired}/{len(rb_results)} tasks")
        if n_fired:
            print(f"  Agreement (predictions match): {n_agree}/{n_fired}")
            print(f"  Both exact:           {n_both:>4}")
            print(f"  Rule only exact:      {n_rb_only:>4}  ← rule-based adds value")
            print(f"  Neural only exact:    {n_nn_only:>4}  ← rule-based hurts")
            print(f"  Neither exact:        {n_fired - n_both - n_rb_only - n_nn_only:>4}")

            if n_rb_only > 0:
                print(f"\n  Tasks where rule-based is correct but neural is not:")
                for r in fired:
                    if r["rule_cmp"]["rule_test_exact"] and not r["rule_cmp"]["neural_test_exact"]:
                        print(f"    {r['task_id']}  neural_acc={r['rule_cmp']['neural_test_acc']:.3f}")
            if n_nn_only > 0:
                print(f"\n  Tasks where neural is correct but rule-based is not:")
                for r in fired:
                    if r["rule_cmp"]["neural_test_exact"] and not r["rule_cmp"]["rule_test_exact"]:
                        print(f"    {r['task_id']}  rule_acc={r['rule_cmp']['rule_test_acc']:.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    mean_ca = float(np.mean([r["cell_acc"]    for r in results]))
    mean_em = float(np.mean([r["exact_match"] for r in results]))
    n_all   = sum(1 for r in results if r["n_exact"] == r["n_pairs"])

    payload = {
        "meta": {
            "checkpoint":  str(ckpt_path),
            "mode":        mode,
            "n_tasks":     n,
            "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "mean_cell_acc":        mean_ca,
            "mean_exact_match":     mean_em,
            "n_tasks_all_exact":    n_all,
            "severity_counts":      sev_counts,
            "severity_pct":         {s: round(100 * c / n, 1)
                                     for s, c in sev_counts.items()},
        },
        # Tasks sorted worst → best so the most interesting ones are at the top
        "tasks": sorted(results, key=lambda r: r["cell_acc"]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}\n")


# ---------------------------------------------------------------------------
# Majority vote helper
# ---------------------------------------------------------------------------

def majority_vote(preds: list[np.ndarray]) -> np.ndarray:
    """Per-cell majority vote over a list of same-shape uint8 grids."""
    stack = np.stack(preds, axis=0)   # (N, H, W)
    votes = np.zeros((10, *stack.shape[1:]), dtype=np.int32)
    for v in range(10):
        votes[v] = (stack == v).sum(axis=0)
    return votes.argmax(axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# D4 geometric symmetry group helpers (for geometric TTA)
# ---------------------------------------------------------------------------

# 8 orientations: 4 rotations × 2 reflections (identity = k_rot=0, flip=False)
_D4 = [(k, f) for f in (False, True) for k in range(4)]


def _d4_apply(grid: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    g = np.rot90(grid, k_rot)
    return np.fliplr(g) if flip else g


def _d4_reverse(grid: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    """Undo a D4 transform applied by _d4_apply with the same (k_rot, flip)."""
    if flip:
        grid = np.fliplr(grid)
    return np.rot90(grid, -k_rot % 4)


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------

def greedy_decode(
    model:   ArcTransformer,
    tok:     ArcTokenizer,
    ctx:     list[tuple[np.ndarray, np.ndarray]],
    test_in: np.ndarray,
    H: int, W: int,
    device:  torch.device,
) -> np.ndarray:
    prefix, pad_mask, gnum = build_prefix(tok, ctx, test_in)
    with torch.no_grad():
        return model.generate(prefix.to(device), pad_mask.to(device), H, W, gnum)


# ---------------------------------------------------------------------------
# TTA: colour-permutation test-time augmentation
# ---------------------------------------------------------------------------

def tta_decode(
    model:   ArcTransformer,
    tok:     ArcTokenizer,
    ctx:     list[tuple[np.ndarray, np.ndarray]],
    test_in: np.ndarray,
    H: int, W: int,
    n_perms: int,
    device:  torch.device,
    rng:     np.random.Generator,
    n_d4:    int = 8,
) -> np.ndarray:
    """Run model with D4 geometric TTA × colour permutations, majority vote.

    For each of n_d4 geometric orientations (from the D4 group), the transform is
    applied to all context and test grids.  Within each orientation, n_perms
    colour-permuted copies are generated.  All predictions are reverse-transformed
    and majority-voted in the original coordinate frame.

    Total evaluations: n_d4 × n_perms.  Set n_d4=1 for colour-only TTA (identity
    orientation only); n_d4=8 uses all D4 symmetries.  Colour 0 (background) is
    never remapped.  90°/270° rotations correctly swap H ↔ W for generation.
    """
    preds = []
    for k_rot, do_flip in _D4[:n_d4]:
        # Apply D4 transform to all context and test grids
        d4_ctx     = [(_d4_apply(inp, k_rot, do_flip),
                       _d4_apply(out, k_rot, do_flip)) for inp, out in ctx]
        d4_test_in = _d4_apply(test_in, k_rot, do_flip)
        # Output dims in transformed space — 90°/270° swaps H ↔ W
        H_t = W if k_rot % 2 else H
        W_t = H if k_rot % 2 else W

        for _ in range(n_perms):
            perm     = np.arange(10, dtype=np.uint8)
            perm[1:] = rng.permutation(9) + 1
            inv_perm = np.zeros(10, dtype=np.uint8)
            for i, v in enumerate(perm):
                inv_perm[v] = i

            perm_ctx = [(perm[inp], perm[out]) for inp, out in d4_ctx]
            pred_t   = greedy_decode(model, tok, perm_ctx, perm[d4_test_in],
                                     H_t, W_t, device)
            pred     = _d4_reverse(inv_perm[pred_t], k_rot, do_flip)
            preds.append(pred)

    return majority_vote(preds)


# ---------------------------------------------------------------------------
# TTT: test-time training
# ---------------------------------------------------------------------------

def _ttt_loo_score(
    model:       ArcTransformer,
    tok:         ArcTokenizer,
    train_pairs: list[dict],
    k_ctx:       int,
    device:      torch.device,
) -> float:
    """Mean cell accuracy over all LOO configs with no augmentation.

    Used during TTT to track the best intermediate model state.
    Skipped when train_pairs has only one pair (nothing to hold out).
    """
    n = len(train_pairs)
    accs = []
    with torch.no_grad():
        for hi in range(n):
            ctx_raw = [p for i, p in enumerate(train_pairs) if i != hi]
            if not ctx_raw:
                continue
            tp      = train_pairs[hi]
            test_in = np.array(tp["input"],  dtype=np.uint8)
            target  = np.array(tp["output"], dtype=np.uint8)
            H, W    = target.shape
            ctx     = [(np.array(p["input"],  dtype=np.uint8),
                        np.array(p["output"], dtype=np.uint8))
                       for p in ctx_raw[:k_ctx]]
            pred = greedy_decode(model, tok, ctx, test_in, H, W, device)
            accs.append(float((pred == target).mean()))
    return float(np.mean(accs)) if accs else 0.0


def ttt_fine_tune(
    model:       ArcTransformer,
    tok:         ArcTokenizer,
    train_pairs: list[dict],   # [{input, output}, ...]  ← ARC dict format
    n_steps:     int,
    lr:          float,
    device:      torch.device,
    rng:         np.random.Generator,
    k_ctx:       int = 3,
    batch_size:  int = 8,
    eval_every:  int = 20,
) -> ArcTransformer:
    """Return a fine-tuned *copy* of model (the original model is not modified).

    Improvements over naive TTT:
    - Batch size > 1: each step trains on `batch_size` differently-augmented
      versions of the same LOO configuration, giving much richer gradient signal.
    - Cosine LR decay: starts at `lr`, tapers to lr/100 by the final step.
    - Save-best: every `eval_every` steps, evaluate clean LOO cell accuracy and
      keep the model state that achieved the highest score. Guards against the
      overfitting cliff that occurs with few training pairs.
    """
    ttt = copy.deepcopy(model)
    ttt.train()
    opt       = torch.optim.AdamW(ttt.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=lr * 0.01
    )
    n         = len(train_pairs)
    can_eval  = (n >= 2)   # need ≥2 pairs for meaningful LOO eval

    best_score = -1.0
    best_state = copy.deepcopy(ttt.state_dict())

    for step in range(n_steps):
        # Pick one LOO configuration for this step
        test_i  = int(rng.integers(n))
        ctx_idx = [i for i in range(n) if i != test_i]
        if len(ctx_idx) > k_ctx:
            ctx_idx = rng.choice(ctx_idx, size=k_ctx, replace=False).tolist()

        # Build batch_size augmented variants of this LOO configuration
        batch_feats:  list[np.ndarray] = []
        batch_lmasks: list[np.ndarray] = []

        for _ in range(batch_size):
            # Collect grids
            grids = []
            for i in ctx_idx:
                grids.append(np.array(train_pairs[i]["input"],  dtype=np.uint8))
                grids.append(np.array(train_pairs[i]["output"], dtype=np.uint8))
            grids.append(np.array(train_pairs[test_i]["input"],  dtype=np.uint8))
            grids.append(np.array(train_pairs[test_i]["output"], dtype=np.uint8))

            # D4 geometric augmentation (same transform applied to all grids)
            k_rot   = int(rng.integers(4))
            do_flip = rng.random() < 0.5
            aug = [np.rot90(g, k_rot) for g in grids]
            if do_flip:
                aug = [np.fliplr(g) for g in aug]

            # Colour permutation (background 0 stays fixed)
            perm     = np.arange(10, dtype=np.uint8)
            perm[1:] = rng.permutation(9) + 1
            aug = [perm[g] for g in aug]

            nc    = len(ctx_idx)
            ctx_p = [(aug[2*j], aug[2*j+1]) for j in range(nc)]
            ti_g  = aug[2*nc]
            to_g  = aug[2*nc + 1]

            feats, lmask = tok.encode_sequence(ctx_p, ti_g, to_g)
            if lmask.any():
                batch_feats.append(feats)
                batch_lmasks.append(lmask)

        if not batch_feats:
            scheduler.step()
            continue

        # Stack into a padded batch (sequences within one task are the same
        # length, so padding is zero or minimal)
        max_len = max(f.shape[0] for f in batch_feats)
        B       = len(batch_feats)
        ft_batch  = torch.zeros(B, max_len, 5, dtype=torch.long,  device=device)
        lm_batch  = torch.zeros(B, max_len,    dtype=torch.bool,  device=device)
        pad_batch = torch.ones( B, max_len,    dtype=torch.bool,  device=device)

        for b, (feats, lmask) in enumerate(zip(batch_feats, batch_lmasks)):
            L = feats.shape[0]
            ft_batch[b, :L]  = torch.from_numpy(feats).long()
            lm_batch[b, :L]  = torch.from_numpy(lmask)
            pad_batch[b, :L] = False

        logits = ttt(ft_batch, pad_batch)
        sl  = logits[:, :-1].contiguous()
        st  = ft_batch[:, 1:, 0].contiguous()
        sm  = lm_batch[:, 1:].contiguous()
        fl  = sl.reshape(-1, sl.size(-1))
        ft2 = st.reshape(-1)
        fm  = sm.reshape(-1)

        if not fm.any():
            scheduler.step()
            continue

        loss = F.cross_entropy(fl[fm], ft2[fm].long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ttt.parameters(), 1.0)
        opt.step()
        scheduler.step()

        # Save-best: check LOO accuracy periodically
        if can_eval and (step + 1) % eval_every == 0:
            ttt.eval()
            score = _ttt_loo_score(ttt, tok, train_pairs, k_ctx, device)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(ttt.state_dict())
            ttt.train()

    # Restore the best observed state
    ttt.load_state_dict(best_state)
    ttt.eval()
    return ttt


def ttt_decode(
    model:       ArcTransformer,
    tok:         ArcTokenizer,
    ctx_raw:     list[dict],    # context pairs in ARC dict format (no held-out pair)
    test_in:     np.ndarray,
    H: int, W:   int,
    n_steps:     int,
    n_perms:     int,
    lr:          float,
    device:      torch.device,
    rng:         np.random.Generator,
    k_ctx:       int = 3,
    batch_size:  int = 8,
    eval_every:  int = 20,
    n_d4:        int = 8,
) -> np.ndarray:
    """Fine-tune on available context pairs, then run TTA for the prediction."""
    ttt = ttt_fine_tune(model, tok, ctx_raw, n_steps, lr, device, rng,
                        k_ctx, batch_size, eval_every)
    ctx = [(np.array(p["input"],  dtype=np.uint8),
            np.array(p["output"], dtype=np.uint8))
           for p in ctx_raw[:k_ctx]]
    if n_perms > 1:
        return tta_decode(ttt, tok, ctx, test_in, H, W, n_perms, device, rng, n_d4)
    return greedy_decode(ttt, tok, ctx, test_in, H, W, device)


# ---------------------------------------------------------------------------
# Per-task evaluation (leave-one-out over training pairs)
# ---------------------------------------------------------------------------

def evaluate_task(
    task:                dict,
    model:               ArcTransformer,
    tok:                 ArcTokenizer,
    mode:                str,
    n_perms:             int,
    ttt_steps:           int,
    ttt_lr:              float,
    device:              torch.device,
    rng:                 np.random.Generator,
    k_ctx:               int,
    verbose:             bool,
    ttt_batch_size:      int  = 8,
    ttt_eval_every:      int  = 20,
    n_d4:                int  = 8,
    analyze:             bool = False,
    compare_rule_based:  bool = False,
) -> dict:
    train = task["train"]
    accs, exacts = [], []
    pair_details: list[dict] = []

    for hi in range(len(train)):
        ctx_raw = [p for i, p in enumerate(train) if i != hi]
        tp      = train[hi]
        test_in = np.array(tp["input"],  dtype=np.uint8)
        target  = np.array(tp["output"], dtype=np.uint8)
        H, W    = target.shape
        ctx     = [(np.array(p["input"],  dtype=np.uint8),
                    np.array(p["output"], dtype=np.uint8))
                   for p in ctx_raw[:k_ctx]]

        if mode == "greedy":
            pred = greedy_decode(model, tok, ctx, test_in, H, W, device)
        elif mode == "tta":
            pred = tta_decode(model, tok, ctx, test_in, H, W, n_perms, device, rng, n_d4)
        else:   # ttt
            pred = ttt_decode(model, tok, ctx_raw, test_in, H, W,
                              ttt_steps, n_perms, ttt_lr, device, rng,
                              k_ctx, ttt_batch_size, ttt_eval_every, n_d4)

        ca = float((pred == target).mean())
        em = bool(np.array_equal(pred, target))
        accs.append(ca)
        exacts.append(em)

        if verbose:
            print(f"    [{hi}] cell_acc={ca:.3f}  exact={'YES' if em else 'no'}")

        if analyze:
            wrong_mask = pred != target
            n_wrong    = int(wrong_mask.sum())
            # Colour confusion for wrong cells: list of [true_colour, pred_colour]
            errors = []
            if n_wrong:
                true_wrong = target[wrong_mask].tolist()
                pred_wrong = pred[wrong_mask].tolist()
                errors = [[t, p] for t, p in zip(true_wrong, pred_wrong)]
            pair_details.append({
                "pair_idx": hi,
                "cell_acc": round(ca, 4),
                "exact":    em,
                "n_wrong":  n_wrong,
                "n_cells":  int(H * W),
                "output_h": H,
                "output_w": W,
                "errors":   errors,
            })

    mean_ca = float(np.mean(accs))
    result  = {
        "task_id":     task["task_id"],
        "cell_acc":    mean_ca,
        "exact_match": float(np.mean([int(e) for e in exacts])),
        "n_pairs":     len(train),
        "n_exact":     sum(exacts),
    }

    if analyze:
        first = pair_details[0] if pair_details else {}
        result.update({
            "severity":     _severity(mean_ca),
            "cell_acc_std": float(np.std(accs)),
            "mean_n_wrong": float(np.mean([p["n_wrong"] for p in pair_details])),
            "output_h":     first.get("output_h", 0),
            "output_w":     first.get("output_w", 0),
            "pairs":        pair_details,
        })

    # ── Test-pair comparison: neural (greedy) vs rule-based solver ────────────
    # Both methods are run on the actual test pair using all training pairs as
    # context.  This is independent of the LOO evaluation above and gives a
    # direct head-to-head on genuinely unseen data.
    if compare_rule_based:
        test_pairs = task.get("test", [])
        ctx_all    = [(np.array(p["input"],  dtype=np.uint8),
                       np.array(p["output"], dtype=np.uint8))
                      for p in train[:k_ctx]]
        rule_pred  = try_rule_based(task)   # None if no solver fires

        neural_test_accs:   list[float] = []
        neural_test_exacts: list[bool]  = []
        rule_test_accs:     list[float] = []
        rule_test_exacts:   list[bool]  = []
        agreements:         list[bool]  = []

        for tp in test_pairs:
            if "output" not in tp:
                continue
            test_in = np.array(tp["input"],  dtype=np.uint8)
            target  = np.array(tp["output"], dtype=np.uint8)
            H, W    = target.shape

            # Neural greedy on actual test pair
            neural_pred = greedy_decode(model, tok, ctx_all, test_in, H, W, device)
            n_ca = float((neural_pred == target).mean())
            n_em = bool(np.array_equal(neural_pred, target))
            neural_test_accs.append(n_ca)
            neural_test_exacts.append(n_em)

            # Rule-based (already computed for whole task)
            if rule_pred is not None and rule_pred.shape == target.shape:
                r_ca = float((rule_pred == target).mean())
                r_em = bool(np.array_equal(rule_pred, target))
                rule_test_accs.append(r_ca)
                rule_test_exacts.append(r_em)
                agreements.append(bool(np.array_equal(neural_pred, rule_pred)))

        cmp: dict = {
            "neural_test_acc":   float(np.mean(neural_test_accs))   if neural_test_accs   else None,
            "neural_test_exact": bool(all(neural_test_exacts))       if neural_test_exacts else None,
            "rule_fires":        rule_pred is not None,
            "rule_test_acc":     float(np.mean(rule_test_accs))      if rule_test_accs     else None,
            "rule_test_exact":   bool(all(rule_test_exacts))         if rule_test_exacts   else None,
            "agree":             bool(all(agreements))               if agreements         else None,
        }

        if verbose and rule_pred is not None:
            tag = "AGREE" if cmp["agree"] else "DISAGREE"
            print(f"    [test] neural={cmp['neural_test_acc']:.3f}{'✓' if cmp['neural_test_exact'] else '✗'}"
                  f"  rule={cmp['rule_test_acc']:.3f}{'✓' if cmp['rule_test_exact'] else '✗'}"
                  f"  {tag}")
        elif verbose:
            print(f"    [test] neural={cmp['neural_test_acc']:.3f}{'✓' if cmp['neural_test_exact'] else '✗'}"
                  f"  rule=n/a")

        result["rule_cmp"] = cmp

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate an ARC transformer checkpoint with greedy / TTA / TTT inference."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to .pt checkpoint (e.g. checkpoints/transformer_cC8_compartment_fill_best.pt)")
    ap.add_argument("--mode",       default="all",
                    choices=["greedy", "tta", "ttt", "all"],
                    help="Inference mode (default: all — runs all three for comparison)")
    ap.add_argument("--n-perms",    type=int,   default=20,
                    help="Number of colour permutations per D4 orientation for TTA (default: 20)")
    ap.add_argument("--n-d4",       type=int,   default=8,
                    help="D4 orientations for geometric TTA: 1=colour-only, 8=all D4 symmetries (default: 8)")
    ap.add_argument("--ttt-steps",      type=int,   default=200,
                    help="Fine-tuning steps for TTT (default: 200)")
    ap.add_argument("--ttt-lr",         type=float, default=1e-4,
                    help="Learning rate for TTT fine-tuning (default: 1e-4)")
    ap.add_argument("--ttt-batch-size", type=int,   default=8,
                    help="Augmented sequences per TTT step (default: 8)")
    ap.add_argument("--ttt-eval-every", type=int,   default=20,
                    help="Steps between save-best LOO evaluations (default: 20)")
    ap.add_argument("--k-context",  type=int,   default=3,
                    help="Max context pairs at inference / TTT fine-tune (default: 3)")
    ap.add_argument("--task-ids",   nargs="+",  default=None,
                    help="Override task IDs (default: read from checkpoint)")
    ap.add_argument("--verbose",    action="store_true",
                    help="Print per-pair breakdown")
    ap.add_argument("--analyze",    action="store_true",
                    help="Collect per-pair error detail and print a ranked analysis report")
    ap.add_argument("--compare-rule-based", action="store_true",
                    help="For each task, also run rule-based solvers on the actual test pair "
                         "and compare with the neural model prediction")
    ap.add_argument("--output-file", default=None,
                    help="Where to save the JSON analysis (default: results/error_analysis_{mode}_{ckpt_stem}.json)")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}\n")

    model, saved_args, ckpt_ids = load_checkpoint(args.checkpoint, device)
    task_ids = args.task_ids or ckpt_ids
    if not task_ids:
        print("ERROR: no task IDs found in checkpoint — use --task-ids")
        sys.exit(1)

    tasks = []
    for tid in task_ids:
        p = DATA_DIR / f"{tid}.json"
        if not p.exists():
            print(f"  WARNING: {tid}.json not found — skipping")
            continue
        t = json.loads(p.read_text())
        t["task_id"] = tid
        tasks.append(t)

    print(f"\nEvaluating {len(tasks)} tasks  ({sum(len(t['train']) for t in tasks)} leave-one-out queries total)\n")

    tok = ArcTokenizer()
    rng = np.random.default_rng(args.seed)

    modes = ["greedy", "tta", "ttt"] if args.mode == "all" else [args.mode]
    summary = {}

    for mode in modes:
        hdr = f" {mode.upper()}"
        if mode in ("tta", "ttt"):
            hdr += f"  n_perms={args.n_perms}  n_d4={args.n_d4}"
        if mode == "ttt":
            hdr += (f"  steps={args.ttt_steps}  lr={args.ttt_lr}"
                    f"  batch={args.ttt_batch_size}  eval_every={args.ttt_eval_every}")
        print("=" * 60)
        print(hdr)
        print("=" * 60)
        t0 = time.time()
        results = []

        for task in tasks:
            if args.verbose:
                print(f"\n  Task {task['task_id']}  ({len(task['train'])} pairs):")
            r = evaluate_task(
                task, model, tok, mode,
                args.n_perms, args.ttt_steps, args.ttt_lr,
                device, rng, args.k_context, args.verbose,
                args.ttt_batch_size, args.ttt_eval_every, args.n_d4,
                analyze=args.analyze,
                compare_rule_based=args.compare_rule_based,
            )
            results.append(r)
            print(f"  {r['task_id']}  "
                  f"cell_acc={r['cell_acc']:.3f}  "
                  f"exact_match={r['exact_match']:.3f}  "
                  f"({r['n_exact']}/{r['n_pairs']} exact)")

        elapsed  = time.time() - t0
        mean_ca  = float(np.mean([r["cell_acc"]    for r in results]))
        mean_em  = float(np.mean([r["exact_match"] for r in results]))
        n_all_em = sum(1 for r in results if r["n_exact"] == r["n_pairs"])

        print(f"\n  Mean cell acc:    {mean_ca:.3f}")
        print(f"  Mean exact match: {mean_em:.3f}")
        print(f"  Tasks all-exact:  {n_all_em}/{len(tasks)}")
        print(f"  Time: {elapsed:.0f}s\n")
        summary[mode] = {"cell_acc": mean_ca, "exact_match": mean_em,
                         "n_all_exact": n_all_em}

        if args.analyze:
            ckpt_stem = Path(args.checkpoint).stem
            if args.output_file:
                out_path = Path(args.output_file)
            else:
                out_name = f"error_analysis_{mode}_{ckpt_stem}.json"
                out_path = PROJECT_ROOT / "results" / out_name
            save_analysis(results, mode, args.checkpoint, out_path)

    if len(modes) > 1:
        print("=" * 60)
        print(" COMPARISON")
        print("=" * 60)
        for m, s in summary.items():
            print(f"  {m:8s}  cell_acc={s['cell_acc']:.3f}  "
                  f"exact_match={s['exact_match']:.3f}  "
                  f"all-exact={s['n_all_exact']}/{len(tasks)}")


if __name__ == "__main__":
    main()
