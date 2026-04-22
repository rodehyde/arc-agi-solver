"""
train_transformer.py — Train a cluster-specialist ARC transformer.

Meta-learning setup:
  • Each training step samples one task from the target cluster(s).
  • From that task's 1,000 RE-ARC examples, it draws K context pairs + 1 test.
  • The full sequence is: in1 GRID_SEP out1 GRID_SEP … inK GRID_SEP outK GRID_SEP
                          test_in GRID_SEP test_out
  • Cross-entropy loss is computed only on the test_out tokens.

This trains the transformer to perform in-context learning:
  given K demonstration pairs showing a rule, predict the output for a new input.

Usage:
    # Train specialist for cluster 18 (3× expansion):
    python scripts/train_transformer.py --clusters 18

    # Train on all three PoC clusters:
    python scripts/train_transformer.py --clusters 16 18 26

    # Resume from checkpoint:
    python scripts/train_transformer.py --clusters 18 --resume checkpoints/transformer_c18_epoch_0100.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.arc_tokenizer import ArcTokenizer, VOCAB_SIZE, PAD
from src.transformer_model import ArcTransformer

PROJECT_ROOT = Path(__file__).parent.parent
RE_ARC_DIR   = PROJECT_ROOT / "data" / "re_arc"
CLUSTER_FILE = PROJECT_ROOT / "results" / "cluster_inspection.txt"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"

N_TRAIN = 800
N_VAL   = 200


# ---------------------------------------------------------------------------
# Data loading (same as train_poc.py)
# ---------------------------------------------------------------------------

def get_cluster_task_ids(cluster: int) -> list[str]:
    import re
    text = CLUSTER_FILE.read_text()
    pattern = rf"Cluster {cluster} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Cluster {cluster} not found")
    return re.findall(r"^\s{2}([0-9a-f]{8}):", match.group(1), re.MULTILINE)


def load_task_examples(task_id: str) -> dict:
    path = RE_ARC_DIR / f"{task_id}.json"
    raw = json.load(open(path))
    examples = [
        {"input":  np.array(e["input"],  dtype=np.uint8),
         "output": np.array(e["output"], dtype=np.uint8)}
        for e in raw
    ]
    return {"train": examples[:N_TRAIN], "val": examples[N_TRAIN:]}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_pair(inp: np.ndarray, out: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random D4 symmetry (rotation + optional reflection) to both grids.

    The same transformation is applied to input AND output so the rule is preserved.
    """
    k = int(rng.integers(4))          # 0, 1, 2, 3 × 90°
    inp = np.rot90(inp, k)
    out = np.rot90(out, k)
    if rng.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)
    return inp, out


def augment_color(grids: list[np.ndarray],
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Randomly permute non-zero colors consistently across all grids.

    Color 0 (background/black) is never remapped. Colors 1–9 are shuffled.
    The same permutation is applied to every grid in the list so the rule
    (which is expressed in terms of color relationships) is preserved.
    """
    perm = np.arange(10, dtype=np.uint8)
    shuffle_idx = rng.permutation(9) + 1          # permute colors 1–9
    perm[1:] = shuffle_idx
    return [perm[g] for g in grids]


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def encode_one(
    tokenizer: ArcTokenizer,
    task_data: list[dict],
    ti: int,
    split: str,
    k: int,
    rng: np.random.Generator,
    augment: bool = False,
) -> tuple:
    """Encode a single (context, query) sequence for task index ti.

    If augment=True (training only), apply random D4 geometry + color permutation.
    """
    N = N_TRAIN if split == "train" else N_VAL
    examples = task_data[ti][split]
    idx = rng.choice(N, size=k + 1, replace=False)

    pairs = [(examples[i]["input"].copy(), examples[i]["output"].copy())
             for i in idx[:k + 1]]

    if augment:
        # Same geometric transform for all pairs in the sequence
        k_rot = int(rng.integers(4))
        do_flip = rng.random() < 0.5
        aug_pairs = []
        for inp, out in pairs:
            inp = np.rot90(inp, k_rot)
            out = np.rot90(out, k_rot)
            if do_flip:
                inp = np.fliplr(inp)
                out = np.fliplr(out)
            aug_pairs.append((inp, out))

        # Color permutation across all grids in the sequence
        all_grids = [g for pair in aug_pairs for g in pair]
        all_grids = augment_color(all_grids, rng)
        aug_pairs = [(all_grids[i * 2], all_grids[i * 2 + 1])
                     for i in range(len(aug_pairs))]
        pairs = aug_pairs

    ctx = [(inp, out) for inp, out in pairs[:k]]
    test_inp, test_out = pairs[k]
    return tokenizer.encode_sequence(ctx, test_inp, test_out)


def sample_batch(
    tokenizer: ArcTokenizer,
    task_data: list[dict],
    task_indices: list[int],
    split: str,
    k: int,
    rng: np.random.Generator,
    max_tokens: int = 4000,
) -> dict[str, torch.Tensor]:
    """Token-budget batching: pack as many tasks as fit within max_tokens padded length.

    Given a list of candidate task indices, greedily adds sequences until the
    padded batch size (n_seqs × max_seq_len) would exceed max_tokens.
    A single sequence that exceeds max_tokens is still processed alone.
    """
    sequences = []
    cur_max_len = 0

    aug = (split == "train")
    for ti in task_indices:
        seq = encode_one(tokenizer, task_data, ti, split, k, rng, augment=aug)
        new_max = max(cur_max_len, seq[0].shape[0])  # seq[0] is (T,5) array
        new_total = (len(sequences) + 1) * new_max
        if sequences and new_total > max_tokens:
            break   # budget exceeded — stop adding; process what we have
        sequences.append(seq)
        cur_max_len = new_max

    batch = tokenizer.pad_batch(sequences)
    return {k_: torch.from_numpy(v) for k_, v in batch.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, features: torch.Tensor,
                    loss_mask: torch.Tensor) -> dict[str, float]:
    """Compute cross-entropy loss, cell accuracy, and exact-match rate.

    features: (B, T, 5) — token ids are features[:,:,0]
    """
    tokens = features[:, :, 0]   # (B, T) — colour/special token ids
    # Shift: logits[t] predicts token[t+1]
    shift_logits  = logits[:, :-1, :].contiguous()
    shift_targets = tokens[:, 1:].contiguous()
    shift_mask    = loss_mask[:, 1:].contiguous()

    if not shift_mask.any():
        return {"loss": 0.0, "cell_acc": 0.0, "exact_match": 0.0}

    flat_logits  = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_targets = shift_targets.reshape(-1)
    flat_mask    = shift_mask.reshape(-1)

    loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask].long())

    preds = flat_logits.argmax(dim=-1)
    cell_acc = (preds[flat_mask] == flat_targets[flat_mask]).float().mean().item()

    # Exact match: all loss-masked tokens correct per example
    B = logits.size(0)
    n_exact = 0
    for i in range(B):
        m = shift_mask[i]
        if m.any():
            p = shift_logits[i].argmax(dim=-1)
            if (p[m] == shift_targets[i][m]).all():
                n_exact += 1
    exact_match = n_exact / B

    return {"loss": loss.item(), "cell_acc": cell_acc, "exact_match": exact_match}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="+", type=int, default=[18],
                        help="Cluster IDs to train on")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--steps-per-epoch", type=int,   default=200)
    parser.add_argument("--max-tokens",      type=int,   default=4000,
                        help="Token budget per batch (n_seqs × padded_len ≤ this)")
    parser.add_argument("--k-context",       type=int,   default=3)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--d-model",         type=int,   default=256)
    parser.add_argument("--n-heads",         type=int,   default=8)
    parser.add_argument("--n-layers",        type=int,   default=6)
    parser.add_argument("--max-seq-len",     type=int,   default=2048)
    parser.add_argument("--max-grid-dim",    type=int,   default=32)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--grad-clip",       type=float, default=1.0)
    parser.add_argument("--resume",          default=None)
    parser.add_argument("--log-every",       type=int,   default=5)
    parser.add_argument("--save-every",      type=int,   default=50)
    parser.add_argument("--patience",        type=int,   default=0,
                        help="Stop after this many eval intervals with no val-loss improvement "
                             "(0 = disabled)")
    parser.add_argument("--warmup-epochs",   type=int,   default=3,
                        help="Linear LR warmup epochs before cosine decay (0 = no warmup)")
    parser.add_argument("--log",             default=None)
    args = parser.parse_args()

    if args.log:
        import builtins
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _fh = open(args.log, "a", buffering=1)
        _orig_print = builtins.print
        def _tee(*a, **kw):
            _orig_print(*a, **kw)
            kw.pop("file", None)
            _orig_print(*a, file=_fh, **kw)
        builtins.print = _tee

    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading tasks for clusters {args.clusters}...")
    task_ids = []
    for c in args.clusters:
        task_ids.extend(get_cluster_task_ids(c))
    T = len(task_ids)
    print(f"  {T} tasks")

    print("  Loading RE-ARC examples...", end=" ", flush=True)
    task_data = [load_task_examples(tid) for tid in task_ids]
    print("done")

    # Quick sequence length check on first task
    tokenizer = ArcTokenizer()
    ex = task_data[0]["train"]
    ctx = [(ex[i]["input"], ex[i]["output"]) for i in range(args.k_context)]
    sample_feats, _ = tokenizer.encode_sequence(ctx, ex[3]["input"], ex[3]["output"])
    print(f"  Example sequence length (task 0): {len(sample_feats)} tokens")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ArcTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        max_grid_dim=args.max_grid_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    if args.warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer,
            start_factor=1.0 / args.warmup_epochs,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - args.warmup_epochs, 1),
            eta_min=args.lr / 10,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        no_improve = ckpt.get("no_improve", 0)
        print(f"  Resumed from epoch {ckpt['epoch']}  (best_val_loss={best_val_loss:.4f}, no_improve={no_improve})")

    CKPT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    steps = args.steps_per_epoch
    cluster_tag = "_".join(str(c) for c in args.clusters)

    print(f"\nTraining: {args.epochs} epochs × {steps} steps = {args.epochs*steps:,} total steps")
    print(f"max_tokens={args.max_tokens}, k={args.k_context}, lr={args.lr}, d_model={args.d_model}, layers={args.n_layers}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = ep_acc = ep_exact = 0.0
        ep_batches = 0

        for _ in range(steps):
            # Shuffle all tasks and let token-budget batching decide how many to use
            task_idx = rng.permutation(T).tolist()
            batch = sample_batch(tokenizer, task_data, task_idx, "train",
                                 args.k_context, rng, args.max_tokens)

            features  = batch["features"].to(device).long()    # (B, T, 5) int16→int64
            pad_mask  = batch["pad_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            logits = model(features, pad_mask)           # (B, T, V)
            metrics = compute_metrics(logits, features, loss_mask)

            if metrics["loss"] == 0.0:
                continue   # degenerate batch (no test output tokens) — skip

            shift_logits  = logits[:, :-1, :].contiguous()
            shift_targets = features[:, 1:, 0].contiguous()   # token ids
            shift_mask    = loss_mask[:, 1:].contiguous()
            flat_l = shift_logits.reshape(-1, shift_logits.size(-1))
            flat_t = shift_targets.reshape(-1)
            flat_m = shift_mask.reshape(-1)
            loss = F.cross_entropy(flat_l[flat_m], flat_t[flat_m].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)   # free grad tensors immediately

            ep_loss   += metrics["loss"]
            ep_acc    += metrics["cell_acc"]
            ep_exact  += metrics["exact_match"]
            ep_batches += 1

        # Release MPS cache after every epoch (prevents silent memory leak on Apple Silicon)
        if device.type == "mps":
            torch.mps.empty_cache()

        scheduler.step()

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            model.eval()
            v_loss = v_acc = v_exact = 0.0
            n_vb = 0
            with torch.no_grad():
                # Evaluate each val task individually (avoids padding waste)
                for vi in [[i] for i in range(T)]:
                    batch = sample_batch(tokenizer, task_data, vi, "val",
                                         args.k_context, rng, args.max_tokens)
                    features  = batch["features"].to(device).long()    # int16→int64
                    pad_mask  = batch["pad_mask"].to(device)
                    loss_mask = batch["loss_mask"].to(device)

                    logits = model(features, pad_mask)
                    m = compute_metrics(logits, features, loss_mask)
                    v_loss  += m["loss"]
                    v_acc   += m["cell_acc"]
                    v_exact += m["exact_match"]
                    n_vb += 1

            mean_v_loss = v_loss / n_vb
            improved = mean_v_loss < best_val_loss
            if improved:
                best_val_loss = mean_v_loss
                no_improve = 0
                p_best = CKPT_DIR / f"transformer_c{cluster_tag}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "task_ids": task_ids,
                    "best_val_loss": best_val_loss,
                    "no_improve": no_improve,
                }, p_best)
            else:
                no_improve += 1

            stop_flag = args.patience > 0 and no_improve >= args.patience
            print(
                f"Epoch {epoch+1:04d}/{args.epochs}  "
                f"train loss={ep_loss/max(ep_batches,1):.4f} acc={ep_acc/max(ep_batches,1):.3f} exact={ep_exact/max(ep_batches,1):.3f}  |  "
                f"val loss={mean_v_loss:.4f} acc={v_acc/n_vb:.3f} exact={v_exact/n_vb:.3f}  "
                f"({'BEST ' if improved else f'no_improve={no_improve}/{args.patience} ' if args.patience else ''})"
                f"({time.time()-t0:.1f}s)"
            )
            model.train()

            if stop_flag:
                print(f"Early stopping: val loss did not improve for {args.patience} eval intervals.")
                break

        if (epoch + 1) % args.save_every == 0:
            p = CKPT_DIR / f"transformer_c{cluster_tag}_epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "task_ids": task_ids,
                "best_val_loss": best_val_loss,
                "no_improve": no_improve,
            }, p)
            print(f"  Checkpoint: {p}")

    p = CKPT_DIR / f"transformer_c{cluster_tag}_final.pt"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "task_ids": task_ids,
        "best_val_loss": best_val_loss,
        "no_improve": no_improve,
    }, p)
    print(f"\nTraining complete. Final checkpoint: {p}")


if __name__ == "__main__":
    main()
