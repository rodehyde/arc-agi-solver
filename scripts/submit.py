"""
submit.py — Generate a Kaggle ARC-AGI submission from a trained checkpoint.

Runs inference on tasks in data/evaluation/ (or --data-dir).  For each task:
  Attempt 1: rule-based solver (if it fires), otherwise TTA
  Attempt 2: TTA with a different RNG seed (provides variety vs attempt 1)

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
    python scripts/submit.py --checkpoint checkpoints/transformer_call_400_arc_best.pt

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
    task:     dict,
    model:    ArcTransformer,
    tok:      ArcTokenizer,
    n_perms:  int,
    n_d4:     int,
    k_ctx:    int,
    device:   torch.device,
    rng1:     np.random.Generator,
    rng2:     np.random.Generator,
    verbose:  bool = False,
) -> dict:
    """Return attempt_1, attempt_2 (lists of lists of int) and diagnostic info."""
    train = task["train"]
    tests = task.get("test", [])
    tid   = task.get("task_id", "unknown")

    # Build context from all training pairs (up to k_ctx)
    ctx = [(np.array(p["input"],  dtype=np.uint8),
            np.array(p["output"], dtype=np.uint8))
           for p in train[:k_ctx]]

    # Rule-based attempt (whole-task, once)
    rule_pred = try_rule_based(task)

    attempts_1: list[list[list[int]]] = []
    attempts_2: list[list[list[int]]] = []
    correct_1 = correct_2 = 0
    n_test    = 0

    for tp in tests:
        test_in = np.array(tp["input"], dtype=np.uint8)
        target  = np.array(tp["output"], dtype=np.uint8) if "output" in tp else None

        # Infer output shape
        if target is not None:
            H, W = target.shape
        else:
            H, W = infer_output_shape(task, test_in)

        # Attempt 1: rule-based if it fires and shape matches, else TTA
        if rule_pred is not None and rule_pred.shape == (H, W):
            pred1 = rule_pred
        elif n_perms > 1:
            pred1 = tta_decode(model, tok, ctx, test_in, H, W, n_perms, device, rng1, n_d4)
        else:
            pred1 = greedy_decode(model, tok, ctx, test_in, H, W, device)

        # Attempt 2: TTA with different seed (always neural, gives variety)
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
        rule_tag = "rule✓" if rule_pred is not None else "     "
        acc_str  = f"  exact1={correct_1}/{n_test}  exact2={correct_2}/{n_test}" if n_test else ""
        print(f"  {tid}  {rule_tag}{acc_str}")

    return {
        "task_id":   tid,
        "attempt_1": attempts_1,
        "attempt_2": attempts_2,
        "n_test":    n_test,
        "correct_1": correct_1,  # 0 if no ground truth
        "correct_2": correct_2,
        "rule_fired": rule_pred is not None,
        "has_ground_truth": any("output" in tp for tp in tests),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate Kaggle ARC-AGI submission from a trained checkpoint."
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to .pt checkpoint")
    ap.add_argument("--data-dir",   default=str(EVAL_DIR),
                    help=f"Directory containing ARC task JSON files (default: {EVAL_DIR})")
    ap.add_argument("--n-perms",    type=int, default=20,
                    help="Colour permutations per D4 orientation for TTA (default: 20)")
    ap.add_argument("--n-d4",       type=int, default=8,
                    help="D4 orientations: 1=colour-only, 8=all D4 (default: 8)")
    ap.add_argument("--k-context",  type=int, default=3,
                    help="Max context pairs (default: 3)")
    ap.add_argument("--task-ids",   nargs="+", default=None,
                    help="Restrict to these task IDs (default: all tasks in data-dir)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="RNG seed for attempt_1 TTA (default: 42)")
    ap.add_argument("--seed2",      type=int, default=137,
                    help="RNG seed for attempt_2 TTA (default: 137)")
    ap.add_argument("--output-file", default=None,
                    help="Output JSON path (default: results/submission_{ckpt_stem}.json)")
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # Load checkpoint
    model, saved_args, ckpt_ids = load_checkpoint(args.checkpoint, device)
    tok  = ArcTokenizer()
    rng1 = np.random.default_rng(args.seed)
    rng2 = np.random.default_rng(args.seed2)

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

    print(f"Tasks: {len(tasks)}  |  n_perms={args.n_perms}  n_d4={args.n_d4}  "
          f"k_ctx={args.k_context}")
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print("=" * 60)

    # Run inference
    submission: dict[str, dict] = {}
    results = []
    t0 = time.time()

    for task in tasks:
        r = predict_task(
            task, model, tok,
            n_perms=args.n_perms, n_d4=args.n_d4, k_ctx=args.k_context,
            device=device, rng1=rng1, rng2=rng2, verbose=args.verbose,
        )
        results.append(r)

        # Build submission entry (one prediction per test pair)
        tid = r["task_id"]
        submission[tid] = {}
        for i, (a1, a2) in enumerate(zip(r["attempt_1"], r["attempt_2"])):
            if i == 0:
                submission[tid]["attempt_1"] = a1
                submission[tid]["attempt_2"] = a2
            else:
                # Multiple test pairs: Kaggle format uses attempt_1/attempt_2 per pair
                # For tasks with >1 test pair, append extra keys
                submission[tid][f"attempt_1_{i+1}"] = a1
                submission[tid][f"attempt_2_{i+1}"] = a2

    elapsed = time.time() - t0

    # Accuracy summary (only meaningful when ground truth is available)
    has_gt    = [r for r in results if r["has_ground_truth"]]
    n_gt      = len(has_gt)
    exact1    = sum(r["correct_1"] for r in has_gt)
    exact2    = sum(r["correct_2"] for r in has_gt)
    n_fired   = sum(1 for r in results if r["rule_fired"])
    n_test_total = sum(r["n_test"] for r in results)

    # Per-task exact: task is correct if ALL test pairs correct
    tasks_exact1 = sum(1 for r in has_gt if r["correct_1"] == r["n_test"] and r["n_test"] > 0)
    tasks_exact2 = sum(1 for r in has_gt if r["correct_2"] == r["n_test"] and r["n_test"] > 0)
    # At least one attempt correct
    tasks_either = sum(1 for r in has_gt
                       if (r["correct_1"] == r["n_test"] or r["correct_2"] == r["n_test"])
                       and r["n_test"] > 0)

    print(f"\nTime: {elapsed:.0f}s  ({elapsed/len(tasks):.1f}s/task)")
    print(f"Rule-based fired: {n_fired}/{len(tasks)}")
    if n_gt > 0:
        print(f"\n--- Accuracy (ground truth available for {n_gt}/{len(tasks)} tasks) ---")
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
