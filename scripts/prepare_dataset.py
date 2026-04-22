"""
prepare_dataset.py — Phase 2 data preparation.

Loads RE-ARC examples for specified tasks, pads grids to CANVAS×CANVAS,
stores as uint8 (values 0–9), splits 800/200 per task, and saves to an .npz
file. One-hot encoding happens at training time to keep file sizes small.

Output .npz keys:
  task_ids     — (T,) str array, one entry per task
  cluster_ids  — (T,) int array, cluster label per task
  inputs_train  — (T, 800, H, W) uint8, padded input grids
  outputs_train — (T, 800, H, W) uint8, padded output grids
  inputs_val    — (T, 200, H, W) uint8
  outputs_val   — (T, 200, H, W) uint8

Where H = W = CANVAS (default 40).

Usage:
    python scripts/prepare_dataset.py \\
        --clusters 16 18 26 \\
        --output data/dataset_poc.npz

    python scripts/prepare_dataset.py \\
        --tasks 00d62c1b 9172f3a0 \\
        --cluster-label 0 \\
        --output data/dataset_custom.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RE_ARC_DIR = PROJECT_ROOT / "data" / "re_arc"
CLUSTER_INSPECTION = PROJECT_ROOT / "results" / "cluster_inspection.txt"

CANVAS = 40   # pad all grids to this square canvas
N_TRAIN = 800
N_VAL = 200
assert N_TRAIN + N_VAL == 1000


def pad_grid(grid: list[list[int]], canvas: int = CANVAS) -> np.ndarray:
    """Pad a variable-size grid to (canvas, canvas) with zeros."""
    h, w = len(grid), len(grid[0])
    arr = np.zeros((canvas, canvas), dtype=np.uint8)
    arr[:h, :w] = grid
    return arr


def load_re_arc(task_id: str) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Load 1000 RE-ARC examples for a task.

    Returns:
        inputs, outputs      each (1000, CANVAS, CANVAS) uint8 — padded grids
        in_heights, in_widths   each (1000,) uint8 — actual input dimensions
        out_heights, out_widths each (1000,) uint8 — actual output dimensions
    """
    path = RE_ARC_DIR / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"RE-ARC data not found: {path}")
    examples = json.load(open(path))
    inputs, outputs = [], []
    in_h, in_w, out_h, out_w = [], [], [], []
    for ex in examples:
        ih, iw = len(ex["input"]),  len(ex["input"][0])
        oh, ow = len(ex["output"]), len(ex["output"][0])
        inputs.append(pad_grid(ex["input"]))
        outputs.append(pad_grid(ex["output"]))
        in_h.append(ih); in_w.append(iw)
        out_h.append(oh); out_w.append(ow)
    return (np.stack(inputs), np.stack(outputs),
            np.array(in_h, dtype=np.uint8), np.array(in_w, dtype=np.uint8),
            np.array(out_h, dtype=np.uint8), np.array(out_w, dtype=np.uint8))


def parse_cluster_task_ids(cluster: int) -> list[str]:
    """Extract task IDs for a cluster from the inspection file."""
    import re
    text = CLUSTER_INSPECTION.read_text()
    pattern = rf"Cluster {cluster} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Cluster {cluster} not found in {CLUSTER_INSPECTION}")
    block = match.group(1)
    return re.findall(r"^\s{2}([0-9a-f]{8}):", block, re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clusters", nargs="+", type=int,
                       help="Cluster numbers to include (reads task IDs from cluster_inspection.txt)")
    group.add_argument("--tasks", nargs="+",
                       help="Explicit task IDs to include")
    parser.add_argument("--cluster-label", type=int, default=0,
                        help="Cluster label to assign when using --tasks")
    parser.add_argument("--output", default="data/dataset_poc.npz",
                        help="Output .npz path")
    parser.add_argument("--canvas", type=int, default=CANVAS,
                        help=f"Pad grids to this square size (default {CANVAS})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build task list: [(task_id, cluster_label), ...]
    task_list = []
    if args.clusters:
        for c in args.clusters:
            for tid in parse_cluster_task_ids(c):
                task_list.append((tid, c))
    else:
        for tid in args.tasks:
            task_list.append((tid, args.cluster_label))

    print(f"Tasks to prepare: {len(task_list)}")
    for tid, c in task_list:
        print(f"  {tid} (cluster {c})")

    # Load and split all tasks
    all_task_ids, all_cluster_ids = [], []
    all_in_train, all_out_train = [], []
    all_in_val, all_out_val = [], []
    all_inh_train, all_inw_train, all_outh_train, all_outw_train = [], [], [], []
    all_inh_val,   all_inw_val,   all_outh_val,   all_outw_val   = [], [], [], []

    for tid, cluster_label in task_list:
        print(f"  Loading {tid}...", end=" ", flush=True)
        inputs, outputs, in_h, in_w, out_h, out_w = load_re_arc(tid)

        # Shuffle then split
        idx = rng.permutation(len(inputs))
        tr, va = idx[:N_TRAIN], idx[N_TRAIN:]

        all_task_ids.append(tid)
        all_cluster_ids.append(cluster_label)
        all_in_train.append(inputs[tr]);   all_out_train.append(outputs[tr])
        all_in_val.append(inputs[va]);     all_out_val.append(outputs[va])
        all_inh_train.append(in_h[tr]);    all_inw_train.append(in_w[tr])
        all_outh_train.append(out_h[tr]);  all_outw_train.append(out_w[tr])
        all_inh_val.append(in_h[va]);      all_inw_val.append(in_w[va])
        all_outh_val.append(out_h[va]);    all_outw_val.append(out_w[va])
        print(f"done  (out max {int(out_h.max())}×{int(out_w.max())})")

    def st(lst): return np.stack(lst)

    inputs_train  = st(all_in_train)
    outputs_train = st(all_out_train)
    inputs_val    = st(all_in_val)
    outputs_val   = st(all_out_val)

    T = len(all_task_ids)
    print(f"\nDataset shapes:")
    print(f"  inputs_train:  {inputs_train.shape}  ({inputs_train.nbytes / 1e6:.1f} MB)")
    print(f"  outputs_train: {outputs_train.shape}")
    print(f"  inputs_val:    {inputs_val.shape}")
    print(f"  outputs_val:   {outputs_val.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        task_ids=np.array(all_task_ids),
        cluster_ids=np.array(all_cluster_ids),
        inputs_train=inputs_train,
        outputs_train=outputs_train,
        inputs_val=inputs_val,
        outputs_val=outputs_val,
        # Per-example actual grid dimensions (for masking padding in loss)
        in_heights_train=st(all_inh_train),
        in_widths_train=st(all_inw_train),
        out_heights_train=st(all_outh_train),
        out_widths_train=st(all_outw_train),
        in_heights_val=st(all_inh_val),
        in_widths_val=st(all_inw_val),
        out_heights_val=st(all_outh_val),
        out_widths_val=st(all_outw_val),
    )
    total_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved to {out_path}  ({total_mb:.1f} MB compressed)")
    print(f"  {T} tasks, {N_TRAIN} train + {N_VAL} val examples each")
    print(f"  Canvas: {args.canvas}×{args.canvas}, dtype: uint8 (one-hot at training time)")
    print(f"  Includes per-example output dimensions for loss masking")


if __name__ == "__main__":
    main()
