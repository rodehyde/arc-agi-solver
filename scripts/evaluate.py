"""
evaluate.py — Inference (greedy / TTA / TTT) on a trained ARC checkpoint.

Three inference modes:
  greedy — single forward pass, argmax at each cell position
  tta    — N colour permutations, un-permute predictions, majority-vote per cell
  ttt    — fine-tune on task training pairs for M steps (with augmentation),
            then run TTA on the fine-tuned copy

Evaluation is leave-one-out over the original ARC training pairs for each task:
  for each pair i in task["train"]:
      context = all other training pairs
      query   = pair i (model predicts output from input)
      score   = compare prediction to pair i output

Usage (download checkpoint from Colab first):
    python scripts/evaluate.py --checkpoint checkpoints/transformer_cC8_compartment_fill_best.pt
    python scripts/evaluate.py --checkpoint ... --mode tta   --n-perms 20
    python scripts/evaluate.py --checkpoint ... --mode ttt   --ttt-steps 50 --n-perms 20
    python scripts/evaluate.py --checkpoint ... --mode all           # compare all three
    python scripts/evaluate.py --checkpoint ... --verbose            # per-pair breakdown
"""

import argparse
import copy
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
) -> np.ndarray:
    """Run model N times with random colour permutations, un-permute, majority vote.

    Colour 0 (background) is never remapped.  Colours 1-9 are randomly shuffled,
    consistently across the context pairs and the test input.  Each prediction is
    un-permuted before voting so all votes are in the original colour space.
    """
    preds = []
    for _ in range(n_perms):
        perm     = np.arange(10, dtype=np.uint8)
        perm[1:] = rng.permutation(9) + 1
        inv_perm = np.zeros(10, dtype=np.uint8)
        for i, v in enumerate(perm):
            inv_perm[v] = i

        perm_ctx = [(perm[inp], perm[out]) for inp, out in ctx]
        pred_perm = greedy_decode(model, tok, perm_ctx, perm[test_in], H, W, device)
        preds.append(inv_perm[pred_perm])

    return majority_vote(preds)


# ---------------------------------------------------------------------------
# TTT: test-time training
# ---------------------------------------------------------------------------

def ttt_fine_tune(
    model:       ArcTransformer,
    tok:         ArcTokenizer,
    train_pairs: list[dict],   # [{input, output}, ...]  ← ARC dict format
    n_steps:     int,
    lr:          float,
    device:      torch.device,
    rng:         np.random.Generator,
    k_ctx:       int = 3,
) -> ArcTransformer:
    """Return a fine-tuned *copy* of model (the original model is not modified).

    Fine-tuning uses leave-one-out over train_pairs at each step, with full D4
    geometric augmentation + colour permutation to maximise data diversity.
    """
    ttt = copy.deepcopy(model)
    ttt.train()
    opt = torch.optim.AdamW(ttt.parameters(), lr=lr, weight_decay=1e-2)
    n   = len(train_pairs)

    for _ in range(n_steps):
        # Leave-one-out: pick test pair, sample up to k_ctx context pairs
        test_i  = int(rng.integers(n))
        ctx_idx = [i for i in range(n) if i != test_i]
        if len(ctx_idx) > k_ctx:
            ctx_idx = rng.choice(ctx_idx, size=k_ctx, replace=False).tolist()

        # Collect all grids for joint augmentation
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
        if not lmask.any():
            continue

        ft  = torch.from_numpy(feats).unsqueeze(0).long().to(device)
        lm  = torch.from_numpy(lmask).unsqueeze(0).to(device)
        pad = torch.zeros(1, ft.shape[1], dtype=torch.bool, device=device)

        logits = ttt(ft, pad)
        sl = logits[:, :-1].contiguous()
        st = ft[:, 1:, 0].contiguous()
        sm = lm[:, 1:].contiguous()
        fl = sl.reshape(-1, sl.size(-1))
        ft2 = st.reshape(-1)
        fm  = sm.reshape(-1)
        if not fm.any():
            continue

        loss = F.cross_entropy(fl[fm], ft2[fm].long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ttt.parameters(), 1.0)
        opt.step()

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
) -> np.ndarray:
    """Fine-tune on available context pairs, then run TTA for the prediction."""
    ttt = ttt_fine_tune(model, tok, ctx_raw, n_steps, lr, device, rng, k_ctx)
    ctx = [(np.array(p["input"],  dtype=np.uint8),
            np.array(p["output"], dtype=np.uint8))
           for p in ctx_raw]
    if n_perms > 1:
        return tta_decode(ttt, tok, ctx, test_in, H, W, n_perms, device, rng)
    return greedy_decode(ttt, tok, ctx, test_in, H, W, device)


# ---------------------------------------------------------------------------
# Per-task evaluation (leave-one-out over training pairs)
# ---------------------------------------------------------------------------

def evaluate_task(
    task:      dict,
    model:     ArcTransformer,
    tok:       ArcTokenizer,
    mode:      str,
    n_perms:   int,
    ttt_steps: int,
    ttt_lr:    float,
    device:    torch.device,
    rng:       np.random.Generator,
    k_ctx:     int,
    verbose:   bool,
) -> dict:
    train = task["train"]
    accs, exacts = [], []

    for hi in range(len(train)):
        ctx_raw = [p for i, p in enumerate(train) if i != hi]
        tp      = train[hi]
        test_in = np.array(tp["input"],  dtype=np.uint8)
        target  = np.array(tp["output"], dtype=np.uint8)
        H, W    = target.shape
        ctx     = [(np.array(p["input"],  dtype=np.uint8),
                    np.array(p["output"], dtype=np.uint8))
                   for p in ctx_raw]

        if mode == "greedy":
            pred = greedy_decode(model, tok, ctx, test_in, H, W, device)
        elif mode == "tta":
            pred = tta_decode(model, tok, ctx, test_in, H, W, n_perms, device, rng)
        else:   # ttt
            pred = ttt_decode(model, tok, ctx_raw, test_in, H, W,
                              ttt_steps, n_perms, ttt_lr, device, rng, k_ctx)

        ca = float((pred == target).mean())
        em = bool(np.array_equal(pred, target))
        accs.append(ca)
        exacts.append(em)

        if verbose:
            print(f"    [{hi}] cell_acc={ca:.3f}  exact={'YES' if em else 'no'}")

    return {
        "task_id":     task["task_id"],
        "cell_acc":    float(np.mean(accs)),
        "exact_match": float(np.mean([int(e) for e in exacts])),
        "n_pairs":     len(train),
        "n_exact":     sum(exacts),
    }


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
                    help="Number of colour permutations for TTA (default: 20)")
    ap.add_argument("--ttt-steps",  type=int,   default=50,
                    help="Fine-tuning steps for TTT (default: 50)")
    ap.add_argument("--ttt-lr",     type=float, default=1e-4,
                    help="Learning rate for TTT fine-tuning (default: 1e-4)")
    ap.add_argument("--k-context",  type=int,   default=3,
                    help="Max context pairs at inference / TTT fine-tune (default: 3)")
    ap.add_argument("--task-ids",   nargs="+",  default=None,
                    help="Override task IDs (default: read from checkpoint)")
    ap.add_argument("--verbose",    action="store_true",
                    help="Print per-pair breakdown")
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
            hdr += f"  n_perms={args.n_perms}"
        if mode == "ttt":
            hdr += f"  steps={args.ttt_steps}  lr={args.ttt_lr}"
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
