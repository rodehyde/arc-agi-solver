"""
train_poc.py — Phase 3/4: train the ARC encoder-decoder on the PoC dataset.

Works with ACTUAL grid sizes (no fixed-canvas padding bloat).
Each batch crops examples to their actual content and pads within the batch
to the batch-maximum size. This ensures the gradient signal isn't diluted.

Architecture:
  GridEncoder          — variable H×W one-hot → global 256-dim embedding
  TransformationEncoder — K (input_emb, output_emb) pairs → 256-dim vector
  Decoder              — query spatial features + transform vector → output logits

The transformation vector is trained with two losses:
  reconstruction   masked cross-entropy on predicted output grid cells
  alignment        InfoNCE — transform vector ↔ MiniLM description embedding

Usage:
    python scripts/train_poc.py
    python scripts/train_poc.py --epochs 200 --steps-per-epoch 200 --log logs/train_poc.log
    python scripts/train_poc.py --resume checkpoints/poc_epoch_0050.pt
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
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import ARCSolver, AlignmentProjection, alignment_loss

PROJECT_ROOT   = Path(__file__).parent.parent
RE_ARC_DIR     = PROJECT_ROOT / "data" / "re_arc"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings_refined.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CLUSTER_FILE   = PROJECT_ROOT / "results" / "cluster_inspection.txt"

N_COLOURS = 10
K_CONTEXT = 3
N_TRAIN   = 800
N_VAL     = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cluster_task_ids(cluster: int) -> list[str]:
    import re
    text = CLUSTER_FILE.read_text()
    pattern = rf"Cluster {cluster} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Cluster {cluster} not found in {CLUSTER_FILE}")
    return re.findall(r"^\s{2}([0-9a-f]{8}):", match.group(1), re.MULTILINE)


def load_task_examples(task_id: str) -> dict:
    """Load 1000 RE-ARC examples for a task, split 800/200.

    Returns dict with keys:
        train, val  — each a list of {'input': np.array, 'output': np.array}
    """
    path = RE_ARC_DIR / f"{task_id}.json"
    raw = json.load(open(path))
    examples = [
        {"input":  np.array(e["input"],  dtype=np.uint8),
         "output": np.array(e["output"], dtype=np.uint8)}
        for e in raw
    ]
    return {"train": examples[:N_TRAIN], "val": examples[N_TRAIN:]}


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def pad_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pad a 2D uint8 array to (h, w) with zeros (top-left aligned)."""
    out = np.zeros((h, w), dtype=np.uint8)
    out[:arr.shape[0], :arr.shape[1]] = arr
    return out


def one_hot_batch(arrays: list[np.ndarray]) -> torch.Tensor:
    """One-hot encode a list of same-size 2D uint8 arrays.

    Returns (B, 10, H, W) float32.
    """
    stacked = np.stack(arrays)                          # (B, H, W)
    t = torch.from_numpy(stacked).long()
    oh = F.one_hot(t, num_classes=N_COLOURS).float()    # (B, H, W, 10)
    return oh.permute(0, 3, 1, 2)                       # (B, 10, H, W)


def sample_batch(task_data: list[dict], task_indices: list[int],
                 split: str, k: int, rng: np.random.Generator) -> dict:
    """Sample K context + 1 query per task; pad within batch to batch-max sizes.

    Returns dict:
        context_inputs   (B, K, 10, ih, iw) float32
        context_outputs  (B, K, 10, oh, ow) float32
        query_input      (B, 10, ih, iw) float32
        query_output     (B, oh, ow) int64
        output_mask      (B, oh, ow) bool
        task_indices     list[int]  — which task each batch item belongs to
    """
    B = len(task_indices)
    N = N_TRAIN if split == "train" else N_VAL
    examples_list = [task_data[ti][split] for ti in task_indices]

    # Sample K+1 examples per task
    chosen = [rng.choice(N, size=k + 1, replace=False) for _ in range(B)]
    ctx_in_raw  = [[examples_list[b][i]["input"]  for i in chosen[b][:k]] for b in range(B)]
    ctx_out_raw = [[examples_list[b][i]["output"] for i in chosen[b][:k]] for b in range(B)]
    q_in_raw    = [examples_list[b][chosen[b][k]]["input"]  for b in range(B)]
    q_out_raw   = [examples_list[b][chosen[b][k]]["output"] for b in range(B)]

    # Compute batch-max sizes for input and output separately
    all_in  = [a for row in ctx_in_raw  for a in row] + q_in_raw
    all_out = [a for row in ctx_out_raw for a in row] + q_out_raw
    max_ih = max(a.shape[0] for a in all_in);  max_iw = max(a.shape[1] for a in all_in)
    max_oh = max(a.shape[0] for a in all_out); max_ow = max(a.shape[1] for a in all_out)

    # Pad and stack
    ctx_in_pad  = [[pad_to(ctx_in_raw[b][k_], max_ih, max_iw) for k_ in range(k)] for b in range(B)]
    ctx_out_pad = [[pad_to(ctx_out_raw[b][k_], max_oh, max_ow) for k_ in range(k)] for b in range(B)]

    # Encode: context_inputs (B, K, 10, ih, iw)
    def stack_k(padded_lists):
        # padded_lists: (B, K) list of np arrays (H, W)
        out = []
        for b in range(B):
            oh_b = one_hot_batch(padded_lists[b])  # (K, 10, H, W)
            out.append(oh_b)
        return torch.stack(out)  # (B, K, 10, H, W)

    ctx_in_t  = stack_k(ctx_in_pad)
    ctx_out_t = stack_k(ctx_out_pad)
    q_in_t  = one_hot_batch([pad_to(a, max_ih, max_iw) for a in q_in_raw])  # (B, 10, ih, iw)
    q_out_np = np.stack([pad_to(a, max_oh, max_ow) for a in q_out_raw])     # (B, oh, ow)
    q_out_t  = torch.from_numpy(q_out_np).long()

    # Output mask: True where actual output content is
    mask = torch.zeros(B, max_oh, max_ow, dtype=torch.bool)
    for i, a in enumerate(q_out_raw):
        mask[i, :a.shape[0], :a.shape[1]] = True

    return {
        "context_inputs":  ctx_in_t,
        "context_outputs": ctx_out_t,
        "query_input":     q_in_t,
        "query_output":    q_out_t,
        "output_mask":     mask,
        "task_indices":    task_indices,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    per_cell = F.cross_entropy(logits, targets, reduction="none")
    return per_cell[mask].mean()


def cell_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                  mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds[mask] == targets[mask]).float().mean().item()


def exact_match_rate(logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    B = preds.shape[0]
    n_correct = 0
    for i in range(B):
        m = mask[i]
        if m.any():
            n_correct += int((preds[i][m] == targets[i][m]).all().item())
    return n_correct / B


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="+", type=int, default=[16, 18, 26],
                        help="Cluster IDs to train on")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--k-context", type=int, default=K_CONTEXT)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    if args.log:
        import builtins
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(args.log, "a", buffering=1)
        _orig_print = builtins.print
        def print(*pargs, **kwargs):  # noqa: F811
            _orig_print(*pargs, **kwargs)
            kwargs.pop("file", None)
            _orig_print(*pargs, file=_log_fh, **kwargs)
        builtins.print = print

    # Device
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # Load task data
    print(f"Loading tasks for clusters {args.clusters}...")
    task_ids, cluster_ids = [], []
    for c in args.clusters:
        for tid in get_cluster_task_ids(c):
            task_ids.append(tid)
            cluster_ids.append(c)
    T = len(task_ids)
    print(f"  {T} tasks across {len(args.clusters)} clusters")

    print("  Loading RE-ARC examples...", end=" ", flush=True)
    task_data = [load_task_examples(tid) for tid in task_ids]
    print("done")

    # Load description embeddings
    emb_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    emb_index = {tid: i for i, tid in enumerate(emb_data["task_ids"])}
    desc_embs = np.stack([emb_data["embeddings"][emb_index[tid]] for tid in task_ids])
    desc_embs_t = torch.from_numpy(desc_embs).float().to(device)
    print(f"  Description embeddings: {desc_embs_t.shape}")

    # Models
    model = ARCSolver(
        base_channels=args.base_channels, embed_dim=args.embed_dim
    ).to(device)
    align_proj = AlignmentProjection(
        embed_dim=args.embed_dim, desc_dim=desc_embs.shape[1], proj_dim=args.proj_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_proj   = sum(p.numel() for p in align_proj.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  +  projection: {n_proj:,}")

    all_params = list(model.parameters()) + list(align_proj.parameters())
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        align_proj.load_state_dict(ckpt["align_proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    batch_size = min(args.batch_size, T)
    steps = args.steps_per_epoch

    print(f"\nTraining: {args.epochs} epochs × {steps} steps = "
          f"{args.epochs * steps:,} total gradient steps")
    print(f"batch_size={batch_size}, k={args.k_context}, "
          f"align_weight={args.align_weight}, temp={args.temperature}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        align_proj.train()
        t0 = time.time()

        ep_recon = ep_align = ep_acc = 0.0

        for _ in range(steps):
            task_idx = rng.choice(T, size=batch_size, replace=False).tolist()

            batch = sample_batch(task_data, task_idx, "train",
                                 args.k_context, rng)
            ctx_in  = batch["context_inputs"].to(device)
            ctx_out = batch["context_outputs"].to(device)
            q_in    = batch["query_input"].to(device)
            q_out   = batch["query_output"].to(device)
            mask    = batch["output_mask"].to(device)

            logits, tv = model(ctx_in, ctx_out, q_in)

            # logits output size = query input size (UNet preserves spatial dims)
            # Resize to match actual output size if they differ
            if logits.shape[-2:] != q_out.shape[-2:]:
                logits = F.interpolate(logits, size=q_out.shape[-2:], mode='nearest')

            recon = masked_cross_entropy(logits, q_out, mask)

            td = desc_embs_t[task_idx]
            tv_p, de_p = align_proj(tv, td)
            align = alignment_loss(tv_p, de_p, temperature=args.temperature)

            loss = recon + args.align_weight * align
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            ep_recon += recon.item()
            ep_align += align.item()
            ep_acc   += cell_accuracy(logits.detach(), q_out, mask)

        scheduler.step()

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            model.eval(); align_proj.eval()
            v_recon = v_align = v_acc = v_exact = 0.0
            n_vb = 0
            with torch.no_grad():
                for start in range(0, T, batch_size):
                    vi = list(range(start, min(start + batch_size, T)))
                    batch = sample_batch(task_data, vi, "val",
                                        args.k_context, rng)
                    ctx_in  = batch["context_inputs"].to(device)
                    ctx_out = batch["context_outputs"].to(device)
                    q_in    = batch["query_input"].to(device)
                    q_out   = batch["query_output"].to(device)
                    mask    = batch["output_mask"].to(device)

                    logits, tv = model(ctx_in, ctx_out, q_in)
                    if logits.shape[-2:] != q_out.shape[-2:]:
                        logits = F.interpolate(logits, size=q_out.shape[-2:], mode='nearest')

                    v_recon += masked_cross_entropy(logits, q_out, mask).item()
                    td = desc_embs_t[vi]
                    tv_p, de_p = align_proj(tv, td)
                    v_align += alignment_loss(tv_p, de_p, args.temperature).item()
                    v_acc   += cell_accuracy(logits, q_out, mask)
                    v_exact += exact_match_rate(logits, q_out, mask)
                    n_vb += 1

            print(
                f"Epoch {epoch+1:04d}/{args.epochs}  "
                f"train recon={ep_recon/steps:.4f} align={ep_align/steps:.4f} "
                f"acc={ep_acc/steps:.3f}  |  "
                f"val recon={v_recon/n_vb:.4f} align={v_align/n_vb:.4f} "
                f"acc={v_acc/n_vb:.3f} exact={v_exact/n_vb:.3f}  "
                f"({time.time()-t0:.1f}s)"
            )
            model.train(); align_proj.train()

        if (epoch + 1) % args.save_every == 0:
            p = CHECKPOINT_DIR / f"poc_epoch_{epoch+1:04d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "align_proj": align_proj.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args)}, p)
            print(f"  Checkpoint: {p}")

    p = CHECKPOINT_DIR / "poc_final.pt"
    torch.save({"epoch": args.epochs - 1, "model": model.state_dict(),
                "align_proj": align_proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args)}, p)
    print(f"\nTraining complete. Final checkpoint: {p}")


if __name__ == "__main__":
    main()
