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
from src.arc_tokenizer import (
    ArcTokenizer, VOCAB_SIZE, PAD,
    START_IN, END_IN, START_OUT, END_OUT, ROW_SEP,
    F_TOKEN, F_COL, F_ROW, F_CHANGE, F_GRID,
)
from src.transformer_model import ArcTransformer

PROJECT_ROOT   = Path(__file__).parent.parent
RE_ARC_DIR     = PROJECT_ROOT / "data" / "re_arc"
TOKENIZED_DIR  = PROJECT_ROOT / "data" / "tokenized"
CLUSTER_FILE   = PROJECT_ROOT / "results" / "cluster_inspection.txt"
CATEGORIES_FILE = PROJECT_ROOT / "results" / "categories_training.json"
CKPT_DIR       = PROJECT_ROOT / "checkpoints"

N_TRAIN = 800
N_VAL   = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cluster_task_ids(cluster: int) -> list[str]:
    import re
    text = CLUSTER_FILE.read_text()
    pattern = rf"Cluster {cluster} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Cluster {cluster} not found")
    return re.findall(r"^\s{2}([0-9a-f]{8}):", match.group(1), re.MULTILINE)


def get_category_task_ids(category: str) -> list[str]:
    """Load task IDs for a named category from results/categories_training.json."""
    if not CATEGORIES_FILE.exists():
        raise FileNotFoundError(
            f"Category file not found: {CATEGORIES_FILE}\n"
            "Run:  python -m src.explore  to generate it."
        )
    data = json.load(open(CATEGORIES_FILE))
    task_ids = [tid for tid, cats in data.items() if category in cats]
    if not task_ids:
        raise ValueError(
            f"Category '{category}' not found or has no tasks in {CATEGORIES_FILE}"
        )
    return task_ids


def load_task_examples(task_id: str) -> dict:
    path = RE_ARC_DIR / f"{task_id}.json"
    raw = json.load(open(path))
    examples = [
        {"input":  np.array(e["input"],  dtype=np.uint8),
         "output": np.array(e["output"], dtype=np.uint8)}
        for e in raw
    ]
    return {"train": examples[:N_TRAIN], "val": examples[N_TRAIN:]}


def load_pretokenized(task_id: str) -> dict | None:
    """Load pre-tokenized pair arrays from data/tokenized/<task_id>.npz.

    Returns a dict with keys 'train', 'val', 'train_lens', 'val_lens', or
    None if the file does not exist (caller falls back to live tokenization).
    """
    path = TOKENIZED_DIR / f"{task_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "train":      data["train"],       # (800, max_pair_len, 4) int16
        "val":        data["val"],         # (200, max_pair_len, 4) int16
        "train_lens": data["train_lens"],  # (800,) int32
        "val_lens":   data["val_lens"],    # (200,) int32
    }


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
    pretok_data: list[dict | None] | None = None,
) -> tuple:
    """Encode a single (context, query) sequence for task index ti.

    If pretok_data is provided and the entry for this task is not None, uses
    pre-tokenized pair arrays from disk (fast path).  Otherwise falls back to
    live tokenization (original behaviour, preserved exactly).

    Pre-tokenized fast path:
      - Samples k+1 pair indices from the split.
      - Loads their (T_pair, 4) arrays (token_id, col, row, color_change).
      - Assigns grid_number (1=in1, 2=out1, ..., 2k=test_in, 2k+1=test_out).
      - Applies colour permutation on token_ids 0–9 (background 0 stays 0,
        colours 1–9 are randomly shuffled) — sufficient augmentation without
        geometry transforms.
      - Recomputes color_change for the full assembled sequence in one pass.
      - Returns (features, loss_mask) as (T, 5) int16 and (T,) bool arrays.

    If augment=True on the live path, applies random D4 geometry + color
    permutation (original behaviour unchanged).
    """
    # ------------------------------------------------------------------
    # Fast path: pre-tokenized arrays available
    # ------------------------------------------------------------------
    if pretok_data is not None and pretok_data[ti] is not None:
        pt = pretok_data[ti]
        split_key  = "train"      if split == "train" else "val"
        lens_key   = "train_lens" if split == "train" else "val_lens"
        pairs_arr  = pt[split_key]   # (N, max_pair_len, 4) int16
        pair_lens  = pt[lens_key]    # (N,) int32
        N          = N_TRAIN if split == "train" else N_VAL

        idx = rng.choice(N, size=k + 1, replace=False)

        # Collect unpadded pair arrays: list of (T_pair, 4) int16
        pair_seqs: list[np.ndarray] = [
            pairs_arr[i, : pair_lens[i]].copy() for i in idx
        ]

        # Optional colour permutation on pre-tokenized data.
        # Only colour token_ids (0–9) are remapped; special tokens are left alone.
        if augment:
            perm = np.arange(18, dtype=np.int16)   # identity for all vocab
            shuffle_idx = (rng.permutation(9) + 1).astype(np.int16)
            perm[1:10] = shuffle_idx               # remap colours 1–9; 0 stays 0
            for seq in pair_seqs:
                color_mask = seq[:, F_TOKEN] <= 9
                seq[color_mask, F_TOKEN] = perm[seq[color_mask, F_TOKEN]]

        # Build full (T_total, 5) array:
        #   START | pair0 | pair1 | ... | pair(k-1) | pair(k)[test_in part] END_IN | test_out part
        # Each pair_seq already contains:
        #   START_IN ... END_IN START_OUT ... END_OUT
        # We need to:
        #   1. Assign grid_numbers.
        #   2. Prepend the global START token.
        #   3. Handle test pair specially: keep START_IN…END_IN from pair(k),
        #      then keep START_OUT…END_OUT (those are the test-output tokens, in loss).
        #   4. Append END token.
        #   5. Recompute color_change for the full sequence.

        START_TOK = 11   # arc_tokenizer.START
        END_TOK   = 16   # arc_tokenizer.END

        segments: list[np.ndarray] = []

        # Global START token
        start_row = np.array([[START_TOK, 0, 0, 0, 0]], dtype=np.int16)
        segments.append(start_row)

        # Context pairs: grid_numbers 1=in1, 2=out1, 3=in2, 4=out2, ...
        for pair_idx in range(k):
            seq = pair_seqs[pair_idx]           # (T_pair, 4)
            # Find boundary between input and output halves.
            # The sequence is: START_IN ... END_IN START_OUT ... END_OUT
            # END_IN has token_id == END_IN (13); START_OUT follows it.
            end_in_pos = np.where(seq[:, F_TOKEN] == END_IN)[0]
            if len(end_in_pos) == 0:
                raise ValueError("Malformed pre-tokenized pair: END_IN not found")
            split_pos = end_in_pos[0] + 1      # first token of output half

            in_grid_num  = 2 * pair_idx + 1    # 1, 3, 5, ...
            out_grid_num = 2 * pair_idx + 2    # 2, 4, 6, ...

            in_half  = seq[:split_pos]          # START_IN ... END_IN
            out_half = seq[split_pos:]          # START_OUT ... END_OUT

            in_seg  = np.zeros((len(in_half),  5), dtype=np.int16)
            out_seg = np.zeros((len(out_half), 5), dtype=np.int16)
            in_seg[:,  :4] = in_half
            out_seg[:, :4] = out_half
            in_seg[:,  F_GRID] = in_grid_num
            out_seg[:, F_GRID] = out_grid_num

            segments.append(in_seg)
            segments.append(out_seg)

        # Test pair: last sampled pair
        test_seq     = pair_seqs[k]
        test_in_gnum = 2 * k + 1
        test_ou_gnum = 2 * k + 2

        end_in_pos = np.where(test_seq[:, F_TOKEN] == END_IN)[0]
        if len(end_in_pos) == 0:
            raise ValueError("Malformed pre-tokenized pair: END_IN not found in test pair")
        split_pos = end_in_pos[0] + 1

        test_in_half  = test_seq[:split_pos]   # START_IN ... END_IN
        test_out_half = test_seq[split_pos:]   # START_OUT ... END_OUT

        test_in_seg  = np.zeros((len(test_in_half),  5), dtype=np.int16)
        test_out_seg = np.zeros((len(test_out_half), 5), dtype=np.int16)
        test_in_seg[:,  :4] = test_in_half
        test_out_seg[:, :4] = test_out_half
        test_in_seg[:,  F_GRID] = test_in_gnum
        test_out_seg[:, F_GRID] = test_ou_gnum

        segments.append(test_in_seg)
        segments.append(test_out_seg)

        # Closing END token (grid_number = test_ou_gnum, no spatial pos)
        end_row = np.array([[END_TOK, 0, 0, 0, test_ou_gnum]], dtype=np.int16)
        segments.append(end_row)

        features = np.concatenate(segments, axis=0)  # (T_total, 5)

        # Recompute color_change for entire sequence in one vectorised pass.
        # color_change[t] = 1 iff token[t] is a colour cell (0–9) AND its
        # colour differs from the previous colour cell.  Special tokens and
        # the very first colour cell get change=0.
        is_color = features[:, F_TOKEN] <= 9                          # (T,) bool
        color_positions = np.where(is_color)[0]

        features[:, F_CHANGE] = 0                                     # reset all
        if len(color_positions) > 1:
            prev_cols  = features[color_positions[:-1], F_TOKEN]
            cur_cols   = features[color_positions[1:],  F_TOKEN]
            changed    = (cur_cols != prev_cols).astype(np.int16)
            features[color_positions[1:], F_CHANGE] = changed

        # Build loss_mask: True only for colour/ROW_SEP tokens inside the test
        # output grid (between START_OUT and END_OUT of the test pair, exclusive).
        loss_mask = np.zeros(len(features), dtype=bool)

        # Find the START_OUT token that belongs to test_ou_gnum
        start_out_positions = np.where(
            (features[:, F_TOKEN] == START_OUT) &
            (features[:, F_GRID]  == test_ou_gnum)
        )[0]
        end_out_positions = np.where(
            (features[:, F_TOKEN] == END_OUT) &
            (features[:, F_GRID]  == test_ou_gnum)
        )[0]

        if len(start_out_positions) > 0 and len(end_out_positions) > 0:
            so = int(start_out_positions[0])
            eo = int(end_out_positions[0])
            # Loss on tokens strictly between START_OUT and END_OUT,
            # plus END_OUT itself (matching original tokenizer behaviour).
            loss_mask[so + 1 : eo + 1] = True

        return features, loss_mask

    # ------------------------------------------------------------------
    # Slow path: live tokenization (original behaviour, unchanged)
    # ------------------------------------------------------------------
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
    pretok_data: list[dict | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Token-budget batching: pack as many tasks as fit within max_tokens padded length.

    Given a list of candidate task indices, greedily adds sequences until the
    padded batch size (n_seqs × max_seq_len) would exceed max_tokens.
    A single sequence that exceeds max_tokens is still processed alone.

    If pretok_data is provided, encode_one will use pre-tokenized pair arrays
    for tasks that have them, falling back to live tokenization otherwise.
    """
    sequences = []
    cur_max_len = 0

    aug = (split == "train")
    for ti in task_indices:
        seq = encode_one(tokenizer, task_data, ti, split, k, rng,
                         augment=aug, pretok_data=pretok_data)
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
    parser.add_argument("--clusters", nargs="+", type=int, default=None,
                        help="Cluster IDs to train on (from old clustering results)")
    parser.add_argument("--category", default=None,
                        help="Category name to train on (from results/categories_training.json); "
                             "e.g. STRUCTURE_UNCHANGED.  Takes precedence over --clusters.")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Explicit list of task IDs to train on (e.g. from scene-description "
                             "clusters).  Takes precedence over --category and --clusters.")
    parser.add_argument("--run-name", default=None,
                        help="Override the run tag used for checkpoint filenames. "
                             "Defaults to the category name or cluster IDs.")
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
    if args.task_ids:
        print(f"Loading {len(args.task_ids)} explicit task IDs...")
        task_ids = args.task_ids
        run_tag = "custom_" + "_".join(task_ids[:3]) + (f"_plus{len(task_ids)-3}" if len(task_ids) > 3 else "")
    elif args.category:
        print(f"Loading tasks for category '{args.category}'...")
        task_ids = get_category_task_ids(args.category)
        run_tag = args.category
    elif args.clusters:
        print(f"Loading tasks for clusters {args.clusters}...")
        task_ids = []
        for c in args.clusters:
            task_ids.extend(get_cluster_task_ids(c))
        run_tag = "_".join(str(c) for c in args.clusters)
    else:
        raise ValueError("Specify --task-ids, --category, or --clusters")

    if args.run_name:
        run_tag = args.run_name

    T = len(task_ids)
    print(f"  {T} tasks")

    print("  Loading RE-ARC examples...", end=" ", flush=True)
    task_data = [load_task_examples(tid) for tid in task_ids]
    print("done")

    # Load pre-tokenized pair arrays if available (None for tasks that are missing)
    pretok_data: list[dict | None] = [load_pretokenized(tid) for tid in task_ids]
    n_pretok = sum(1 for p in pretok_data if p is not None)
    if n_pretok > 0:
        print(f"  Pre-tokenized cache: {n_pretok}/{T} tasks (fast path active)")
    else:
        print("  Pre-tokenized cache: none found — using live tokenization")

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
                                 args.k_context, rng, args.max_tokens,
                                 pretok_data=pretok_data)

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
                                         args.k_context, rng, args.max_tokens,
                                         pretok_data=pretok_data)
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
                p_best = CKPT_DIR / f"transformer_c{run_tag}_best.pt"
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
            p = CKPT_DIR / f"transformer_c{run_tag}_epoch_{epoch+1:04d}.pt"
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

    p = CKPT_DIR / f"transformer_c{run_tag}_final.pt"
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
