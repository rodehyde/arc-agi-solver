"""
submit.py — Generate a Kaggle ARC-AGI submission from a trained checkpoint.

Runs inference on tasks in data/evaluation/ (or --data-dir).  For each task:
  Attempt 1: rule-based solver (if it fires), otherwise TTT+TTA or TTA
  Attempt 2: TTA with a different RNG seed (base model, no TTT)

TTT (Test-Time Training) can be enabled with --ttt-steps N.  The model is
fine-tuned on each task's own training pairs before predicting, using all
pairs with augmentation (no leave-one-out).  This significantly improves
accuracy on novel tasks at the cost of ~1–3 min/task extra GPU time.

Caching: completed task results are written to --cache-file after each task
so the run can be interrupted and resumed across Colab sessions.

Output shape is inferred from training pairs when no ground-truth is available.
If the evaluation files include test outputs (as the public ARC-AGI dataset does),
accuracy is computed and printed.

Outputs:
  results/submission_{ckpt_stem}.json        — Kaggle submission JSON
  (accuracy summary printed to stdout)

Kaggle submission format:
  {
    "task_id": {
      "attempt_1": [[row, ...], ...],
      "attempt_2": [[row, ...], ...]
    },
    ...
  }

Usage (local):
    python scripts/submit.py --checkpoint checkpoints/transformer_pooled_278_best.pt

Usage with TTT:
    python scripts/submit.py --checkpoint checkpoints/transformer_pooled_278_best.pt \\
        --ttt-steps 100 --cache-file results/cache_pooled_278.json

Usage (Colab): run via submit_colab.ipynb
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.arc_tokenizer import ArcTokenizer
from src.transformer_model import ArcTransformer
from scripts.evaluate import (
    load_checkpoint,
    greedy_decode,
    tta_decode,
    ttt_decode,
    try_rule_based,
)

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR     = PROJECT_ROOT / "data" / "evaluation"
RESULTS_DIR  = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Output shape inference
# ---------------------------------------------------------------------------

def infer_output_shape(task: dict, test_in: np.ndarray) -> tuple[int, int]:
    """Infer (H, W) of the test output from the task's training pairs.

    Priority:
      1. All training outputs have the same shape → use it.
      2. All training outputs match their paired input shape → use test input shape.
      3. Fallback: most common training output shape.
    """
    train = task["train"]
    out_shapes = [tuple(np.array(p["output"], dtype=np.uint8).shape) for p in train]
    in_shapes  = [tuple(np.array(p["input"],  dtype=np.uint8).shape) for p in train]

    unique_out = set(out_shapes)
    if len(unique_out) == 1:
        return unique_out.pop()  # type: ignore[return-value]

    if out_shapes == in_shapes:
        return tuple(test_in.shape)  # type: ignore[return-value]

    return Counter(out_shapes).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Per-task prediction
# ---------------------------------------------------------------------------

def predict_task(
    task:       dict,
    model:      ArcTransformer,
    tok:        ArcTokenizer,
    n_perms:    int,
    n_d4:       int,
    k_ctx:      int,
    device:     torch.device,
    rng1:       np.random.Generator,
    rng2:       np.random.Generator,
    ttt_steps:  int   = 0,
    ttt_lr:     float = 1e-4,
    verbose:    bool  = False,
) -> dict:
    """Return attempt_1, attempt_2 (lists of lists of int) and diagnostic info.

    Attempt 1: rule-based if it fires, else TTT+TTA (if ttt_steps>0) or TTA.
    Attempt 2: always plain TTA with rng2 (base model, different seed).
    """
    train = task["train"]
    tests = task.get("test", [])
    tid   = task.get("task_id", "unknown")

    # Build context from all training pairs (used as-is for attempt 2 / rule check)
    ctx = [(np.array(p["input"],  dtype=np.uint8),
            np.array(p["output"], dtype=np.uint8))
           for p in train[:k_ctx]]

    # Rule-based attempt (whole-task, once)
    rule_pred = try_rule_based(task)

    # Fine-tune once per task (all pairs, fixed schedule) — reused for every test pair
    if ttt_steps > 0 and rule_pred is None:
        ttt_model_ready = True
    else:
        ttt_model_ready = False

    attempts_1: list[list[list[int]]] = []
    attempts_2: list[list[list[int]]] = []
    correct_1 = correct_2 = 0
    n_test    = 0

    for tp in tests:
        test_in = np.array(tp["input"], dtype=np.uint8)
        target  = np.array(tp["output"], dtype=np.uint8) if "output" in tp else None

        if target is not None:
            H, W = target.shape
        else:
            H, W = infer_output_shape(task, test_in)

        # Attempt 1
        if rule_pred is not None and rule_pred.shape == (H, W):
            pred1 = rule_pred
        elif ttt_model_ready:
            pred1 = ttt_decode(
                model, tok, train, test_in, H, W,
                n_steps=ttt_steps, n_perms=n_perms, lr=ttt_lr,
                device=device, rng=rng1, k_ctx=k_ctx,
                n_d4=n_d4, use_all_pairs=True, fixed_schedule=True,
            )
        elif n_perms > 1:
            pred1 = tta_decode(model, tok, ctx, test_in, H, W, n_perms, device, rng1, n_d4)
        else:
            pred1 = greedy_decode(model, tok, ctx, test_in, H, W, device)

        # Attempt 2: plain TTA on base model, independent seed
        if n_perms > 1:
            pred2 = tta_decode(model, tok, ctx, test_in, H, W, n_perms, device, rng2, n_d4)
        else:
            pred2 = greedy_decode(model, tok, ctx, test_in, H, W, device)

        attempts_1.append(pred1.tolist())
        attempts_2.append(pred2.tolist())
        n_test += 1

        if target is not None:
            correct_1 += int(np.array_equal(pred1, target))
            correct_2 += int(np.array_equal(pred2, target))

    if verbose:
        mode_tag = "ttt " if ttt_model_ready else ("rule" if rule_pred is not None else "tta ")
        acc_str  = f"  exact1={correct_1}/{n_test}  exact2={correct_2}/{n_test}" if n_test else ""
        print(f"  {tid}  [{mode_tag}]{acc_str}")

    return {
        "task_id":         tid,
        "attempt_1":       attempts_1,
        "attempt_2":       attempts_2,
        "n_test":          n_test,
        "correct_1":       correct_1,
        "correct_2":       correct_2,
        "rule_fired":      rule_pred is not None,
        "has_ground_truth": any("output" in tp for tp in tests),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate Kaggle ARC-AGI submission from a trained checkpoint."
    )
    ap.add_argument("--checkpoint",  required=True,
                    help="Path to .pt checkpoint")
    ap.add_argument("--data-dir",    default=str(EVAL_DIR),
                    help=f"Directory containing ARC task JSON files (default: {EVAL_DIR})")
    ap.add_argument("--n-perms",     type=int, default=20,
                    help="Colour permutations per D4 orientation for TTA (default: 20)")
    ap.add_argument("--n-d4",        type=int, default=8,
                    help="D4 orientations: 1=colour-only, 8=all D4 (default: 8)")
    ap.add_argument("--k-context",   type=int, default=3,
                    help="Max context pairs (default: 3)")
    ap.add_argument("--task-ids",    nargs="+", default=None,
                    help="Restrict to these task IDs (default: all tasks in data-dir)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="RNG seed for attempt_1 TTA (default: 42)")
    ap.add_argument("--seed2",       type=int, default=137,
                    help="RNG seed for attempt_2 TTA (default: 137)")
    ap.add_argument("--ttt-steps",   type=int, default=0,
                    help="TTT fine-tuning steps per task (0=disabled, ~100 recommended). "
                         "Fine-tunes on all training pairs before TTA decoding.")
    ap.add_argument("--ttt-lr",      type=float, default=1e-4,
                    help="Learning rate for TTT fine-tuning (default: 1e-4)")
    ap.add_argument("--cache-file",  default=None,
                    help="JSON file to cache completed task results for resume across "
                         "sessions (default: none). Skips already-completed tasks on restart.")
    ap.add_argument("--output-file", default=None,
                    help="Output JSON path (default: results/submission_{ckpt_stem}.json)")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # Load checkpoint
    model, saved_args, ckpt_ids = load_checkpoint(args.checkpoint, device)
    tok  = ArcTokenizer()

    # Load tasks
    data_dir = Path(args.data_dir)
    if args.task_ids:
        task_files = [data_dir / f"{tid}.json" for tid in args.task_ids]
    else:
        task_files = sorted(data_dir.glob("*.json"))

    tasks = []
    for p in task_files:
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping")
            continue
        t = json.loads(p.read_text())
        t["task_id"] = p.stem
        tasks.append(t)

    # Load cache (previously completed results)
    cache_path = Path(args.cache_file) if args.cache_file else None
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Cache: {len(cache)} tasks already completed (loaded from {cache_path})")

    ttt_tag = f"  ttt_steps={args.ttt_steps}  ttt_lr={args.ttt_lr}" if args.ttt_steps else ""
    print(f"Tasks: {len(tasks)}  |  n_perms={args.n_perms}  n_d4={args.n_d4}  "
          f"k_ctx={args.k_context}{ttt_tag}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    n_cached   = sum(1 for t in tasks if t["task_id"] in cache)
    n_remaining = len(tasks) - n_cached
    print(f"Cached: {n_cached}  Remaining: {n_remaining}")
    print("=" * 60)

    results = list(cache.values())
    t0 = time.time()
    n_done = 0

    for task in tasks:
        tid = task["task_id"]
        if tid in cache:
            continue  # already done

        rng1 = np.random.default_rng(args.seed)
        rng2 = np.random.default_rng(args.seed2)

        r = predict_task(
            task, model, tok,
            n_perms=args.n_perms, n_d4=args.n_d4, k_ctx=args.k_context,
            device=device, rng1=rng1, rng2=rng2,
            ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
            verbose=args.verbose,
        )
        results.append(r)
        n_done += 1

        # Write to cache immediately
        if cache_path:
            cache[tid] = r
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.verbose and n_done % 10 == 0:
            elapsed = time.time() - t0
            rate    = elapsed / n_done
            remaining = (n_remaining - n_done) * rate
            print(f"  [{n_done}/{n_remaining}]  {rate:.1f}s/task  "
                  f"ETA {remaining/3600:.1f}h")

    elapsed = time.time() - t0

    # Build submission JSON from all results (cached + new)
    submission: dict[str, dict] = {}
    for r in results:
        tid = r["task_id"]
        submission[tid] = {}
        for i, (a1, a2) in enumerate(zip(r["attempt_1"], r["attempt_2"])):
            if i == 0:
                submission[tid]["attempt_1"] = a1
                submission[tid]["attempt_2"] = a2
            else:
                submission[tid][f"attempt_1_{i+1}"] = a1
                submission[tid][f"attempt_2_{i+1}"] = a2

    # Accuracy summary
    has_gt       = [r for r in results if r["has_ground_truth"]]
    n_gt         = len(has_gt)
    n_fired      = sum(1 for r in results if r["rule_fired"])
    tasks_exact1 = sum(1 for r in has_gt if r["correct_1"] == r["n_test"] and r["n_test"] > 0)
    tasks_exact2 = sum(1 for r in has_gt if r["correct_2"] == r["n_test"] and r["n_test"] > 0)
    tasks_either = sum(1 for r in has_gt
                       if (r["correct_1"] == r["n_test"] or r["correct_2"] == r["n_test"])
                       and r["n_test"] > 0)

    if n_done > 0:
        print(f"\nNew tasks this session: {n_done}  ({elapsed/max(n_done,1):.1f}s/task)")
    print(f"Total tasks in submission: {len(results)}")
    print(f"Rule-based fired: {n_fired}/{len(results)}")
    if n_gt > 0:
        print(f"\n--- Accuracy (ground truth available for {n_gt}/{len(results)} tasks) ---")
        print(f"  Attempt 1:      {tasks_exact1}/{n_gt} tasks exact  "
              f"({100*tasks_exact1/n_gt:.1f}%)")
        print(f"  Attempt 2:      {tasks_exact2}/{n_gt} tasks exact  "
              f"({100*tasks_exact2/n_gt:.1f}%)")
        print(f"  Either correct: {tasks_either}/{n_gt} tasks  "
              f"({100*tasks_either/n_gt:.1f}%)  ← Kaggle score")
    else:
        print("  (No ground truth available — blind submission)")

    # Save submission JSON
    ckpt_stem = Path(args.checkpoint).stem
    out_path  = (Path(args.output_file) if args.output_file
                 else RESULTS_DIR / f"submission_{ckpt_stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)
    print(f"\nSubmission saved: {out_path}  ({len(submission)} tasks)")


if __name__ == "__main__":
    main()
