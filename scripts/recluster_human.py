"""
recluster_human.py — Cluster ARC tasks using human rule descriptions
combined with programmatic I/O features.

The idea: human descriptions capture the *semantic* rule; I/O features capture
*structural* facts (size change, colour counts, fixed output, etc.).
Combining both gives a more discriminative feature vector than either alone.

Usage:
    python scripts/recluster_human.py
    python scripts/recluster_human.py --desc-weight 2.0 --min-cluster-size 3

Outputs:
    results/human_clusters.json        — {task_id: cluster_label} for all annotated tasks
    results/human_cluster_inspection.txt — human-readable cluster report
    results/human_cluster_plot.png      — UMAP scatter coloured by cluster
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan


# ---------------------------------------------------------------------------
# Programmatic feature extraction (mirrors compute_features in annotate_tasks)
# ---------------------------------------------------------------------------

def extract_io_features(task: dict) -> np.ndarray:
    """
    Return a fixed-length float32 vector of I/O structural features.

    Features (12 dimensions):
      [0]  frac_same        — fraction of pairs where output size == input size
      [1]  frac_grow_2x     — fraction where output is exactly 2× input
      [2]  frac_grow_3x     — fraction where output is exactly 3× input
      [3]  frac_shrink      — fraction where output is strictly smaller than input
      [4]  frac_grow_other  — fraction where output larger but not 2× or 3×
      [5]  output_fixed     — 1 if all outputs have the same shape, else 0
      [6]  out_h_norm       — output height / 30  (0 if not fixed)
      [7]  out_w_norm       — output width / 30   (0 if not fixed)
      [8]  in_colours_norm  — mean unique colours in input / 10
      [9]  out_colours_norm — mean unique colours in output / 10
      [10] new_colours      — 1 if any output contains a colour not in its input
      [11] n_pairs_norm     — number of train pairs / 10
    """
    pairs = task["train"]
    in_shapes  = [np.array(p["input"]).shape  for p in pairs]
    out_shapes = [np.array(p["output"]).shape for p in pairs]

    counts = {"same": 0, "grow_2x": 0, "grow_3x": 0, "shrink": 0, "grow_other": 0}
    for (ih, iw), (oh, ow) in zip(in_shapes, out_shapes):
        if (oh, ow) == (ih, iw):
            counts["same"] += 1
        elif oh == ih * 2 and ow == iw * 2:
            counts["grow_2x"] += 1
        elif oh == ih * 3 and ow == iw * 3:
            counts["grow_3x"] += 1
        elif oh < ih or ow < iw:
            counts["shrink"] += 1
        elif oh > ih or ow > iw:
            counts["grow_other"] += 1

    n = len(pairs)
    frac_same       = counts["same"]       / n
    frac_grow_2x    = counts["grow_2x"]    / n
    frac_grow_3x    = counts["grow_3x"]    / n
    frac_shrink     = counts["shrink"]     / n
    frac_grow_other = counts["grow_other"] / n

    fixed_out      = len(set(out_shapes)) == 1
    out_h_norm     = (out_shapes[0][0] / 30.0) if fixed_out else 0.0
    out_w_norm     = (out_shapes[0][1] / 30.0) if fixed_out else 0.0

    in_colors  = [set(np.array(p["input"]).flat)  for p in pairs]
    out_colors = [set(np.array(p["output"]).flat) for p in pairs]

    in_colours_norm  = np.mean([len(c) for c in in_colors])  / 10.0
    out_colours_norm = np.mean([len(c) for c in out_colors]) / 10.0

    new_colours = float(any(oc > ic for oc, ic in zip(out_colors, in_colors)))

    n_pairs_norm = len(pairs) / 10.0

    return np.array([
        frac_same, frac_grow_2x, frac_grow_3x, frac_shrink, frac_grow_other,
        float(fixed_out), out_h_norm, out_w_norm,
        in_colours_norm, out_colours_norm, new_colours, n_pairs_norm,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Cluster inspection report
# ---------------------------------------------------------------------------

def write_inspection(task_ids, descriptions, io_feats, labels, output_path: Path):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    lines = [
        f"Human description cluster inspection",
        f"  {len(task_ids)} tasks   {n_clusters} clusters\n",
    ]

    feat_names = [
        "same-size", "2× grow", "3× grow", "shrink", "other-grow",
        "fixed-out", "out-h/30", "out-w/30",
        "in-colours/10", "out-colours/10", "new-colours", "n-pairs/10",
    ]

    for c in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == c]
        header = f"Cluster {c}" if c >= 0 else "Noise"
        lines += [f"\n{'='*60}", f"{header} (n={len(idxs)})", f"{'='*60}"]

        # Mean I/O feature vector for this cluster
        if len(idxs) > 1:
            mean_feats = np.stack([io_feats[i] for i in idxs]).mean(axis=0)
            lines.append("  I/O features (mean): " +
                         "  ".join(f"{feat_names[j]}={mean_feats[j]:.2f}"
                                   for j in range(len(feat_names))
                                   if mean_feats[j] > 0.05))

        for i in idxs:
            lines.append(f"\n  {task_ids[i]}:")
            lines.append(f"  {descriptions[i]}")

    output_path.write_text("\n".join(lines))
    print(f"  Inspection saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc-file", default="data/human_descriptions.json",
                        help="Human descriptions JSON produced by the annotation tool")
    parser.add_argument("--training-dir", default="data/training",
                        help="Directory of original ARC training JSON files")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model for embedding descriptions")
    parser.add_argument("--desc-weight", type=float, default=1.5,
                        help="Scale factor applied to description embedding relative to I/O features. "
                             "Higher → descriptions dominate. Lower → structure dominates.")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="HDBSCAN min_samples")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output files")
    args = parser.parse_args()

    desc_path    = Path(args.desc_file)
    training_dir = Path(args.training_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load human descriptions ──────────────────────────────────────────────
    if not desc_path.exists():
        print(f"No human descriptions found at {desc_path}.")
        print("Run the annotation notebook first:  jupyter notebook notebooks/annotate_tasks.ipynb")
        return

    human_descs = json.loads(desc_path.read_text())
    if not human_descs:
        print("No descriptions yet — annotate some tasks first.")
        return

    task_ids     = sorted(human_descs.keys())
    descriptions = [human_descs[tid]["description"] for tid in task_ids]
    print(f"Loaded {len(task_ids)} human descriptions.")

    # ── Extract programmatic I/O features ───────────────────────────────────
    print("Extracting I/O features...")
    io_feats_list = []
    missing = []
    for tid in task_ids:
        task_file = training_dir / f"{tid}.json"
        if not task_file.exists():
            missing.append(tid)
            io_feats_list.append(np.zeros(12, dtype=np.float32))
        else:
            task = json.loads(task_file.read_text())
            io_feats_list.append(extract_io_features(task))

    if missing:
        print(f"  Warning: {len(missing)} task files not found: {missing[:5]}...")

    io_feats = np.stack(io_feats_list)  # (N, 12)

    # ── Embed descriptions ───────────────────────────────────────────────────
    print(f"Embedding descriptions with '{args.model}'...")
    embedder = SentenceTransformer(args.model)
    desc_embs = embedder.encode(descriptions, show_progress_bar=True,
                                normalize_embeddings=True)  # (N, 384) unit vectors
    print(f"  Embedding shape: {desc_embs.shape}")

    # ── Build joint feature vector ───────────────────────────────────────────
    # Description embedding (unit-normed) scaled by desc_weight.
    # I/O features are already in [0, 1] per dimension.
    joint = np.concatenate(
        [args.desc_weight * desc_embs, io_feats],
        axis=1,
    )  # (N, 384 + 12)
    print(f"  Joint feature vector: {joint.shape}  (desc×{args.desc_weight} + {io_feats.shape[1]} I/O)")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    print("Running UMAP...")
    n_neighbors = min(15, len(task_ids) - 1)
    reducer = UMAP(n_components=2, random_state=42,
                   n_neighbors=n_neighbors, min_dist=0.1)
    xy = reducer.fit_transform(joint)

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    print(f"Running HDBSCAN(min_cluster_size={args.min_cluster_size}, "
          f"min_samples={args.min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    labels = clusterer.fit_predict(xy)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  {n_clusters} clusters, {n_noise} noise points")
    for c in sorted(set(labels)):
        n = (labels == c).sum()
        tag = f"Cluster {c}" if c >= 0 else "Noise"
        print(f"    {tag}: {n} tasks")

    # ── Save cluster assignments ──────────────────────────────────────────────
    assignments = {tid: int(labels[i]) for i, tid in enumerate(task_ids)}
    out_json = output_dir / "human_clusters.json"
    out_json.write_text(json.dumps(assignments, indent=2))
    print(f"  Cluster assignments saved to {out_json}")

    # ── Inspection report ─────────────────────────────────────────────────────
    write_inspection(
        task_ids, descriptions, [io_feats[i] for i in range(len(task_ids))],
        labels, output_dir / "human_cluster_inspection.txt"
    )

    # ── UMAP scatter plot ─────────────────────────────────────────────────────
    cmap = plt.cm.tab20
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, tid in enumerate(task_ids):
        lbl = labels[i]
        colour = cmap((lbl % 20) / 20) if lbl >= 0 else (0.75, 0.75, 0.75, 1.0)
        ax.scatter(xy[i, 0], xy[i, 1], color=colour, s=60, zorder=3)
        ax.annotate(tid, (xy[i, 0], xy[i, 1]), fontsize=5, ha="left",
                    va="bottom", xytext=(3, 3), textcoords="offset points")

    handles = []
    for c in sorted(set(labels)):
        n = (labels == c).sum()
        colour = cmap((c % 20) / 20) if c >= 0 else (0.75, 0.75, 0.75, 1.0)
        tag = f"Cluster {c} (n={n})" if c >= 0 else f"Noise (n={n})"
        handles.append(mpatches.Patch(color=colour, label=tag))
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              ncol=max(1, n_clusters // 15), framealpha=0.7)

    ax.set_title(f"Human description clustering — {len(task_ids)} tasks, "
                 f"{n_clusters} clusters\n"
                 f"(desc_weight={args.desc_weight}, "
                 f"min_cluster_size={args.min_cluster_size})")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    plt.tight_layout()
    out_plot = output_dir / "human_cluster_plot.png"
    plt.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_plot}")

    print("\nDone.")


if __name__ == "__main__":
    main()
