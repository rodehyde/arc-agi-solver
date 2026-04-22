"""
recluster_large.py — Re-cluster the large cluster (default: cluster 4) using
HDBSCAN with looser parameters and agglomerative clustering.

Usage:
    python scripts/recluster_large.py
    python scripts/recluster_large.py --source-cluster 4 --agg-k 12
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster
from umap import UMAP
import hdbscan


def parse_cluster_task_ids(inspection_path: Path, cluster: int) -> list[str]:
    text = inspection_path.read_text()
    pattern = rf"Cluster {cluster} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    return re.findall(r"^\s{2}([0-9a-f]{8}):", block, re.MULTILINE)


def plot_clusters(xy, labels, task_ids, title, output_path):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(16, 11))
    for i, tid in enumerate(task_ids):
        lbl = labels[i]
        colour = cmap((lbl % 20) / 20) if lbl >= 0 else (0.75, 0.75, 0.75, 1.0)
        ax.scatter(xy[i, 0], xy[i, 1], color=colour, s=50, zorder=3)

    # Legend
    cluster_ids = sorted(set(labels))
    handles = []
    for c in cluster_ids:
        n = (labels == c).sum()
        colour = cmap((c % 20) / 20) if c >= 0 else (0.75, 0.75, 0.75, 1.0)
        label = f"Cluster {c} (n={n})" if c >= 0 else f"Noise (n={n})"
        handles.append(mpatches.Patch(color=colour, label=label))
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              ncol=max(1, n_clusters // 15), framealpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_path}")


def write_inspection(task_ids, descriptions, labels, output_path: Path):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    lines = [f"Sub-cluster inspection — {len(task_ids)} tasks, {n_clusters} sub-clusters\n"]

    for c in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == c]
        header = f"Sub-cluster {c}" if c >= 0 else "Noise"
        lines += [f"\n{'='*60}", f"{header} (n={len(idxs)})", f"{'='*60}"]
        for i in idxs:
            lines += [f"\n  {task_ids[i]}:", f"  {descriptions[i]}"]

    output_path.write_text("\n".join(lines))
    print(f"  Inspection saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cluster", type=int, default=4)
    parser.add_argument("--inspection", default="results/cluster_inspection.txt")
    parser.add_argument("--embeddings", default="data/embeddings_training.npz")
    parser.add_argument("--output-dir", default="results/cluster_4_split")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=3)
    parser.add_argument("--hdbscan-min-samples", type=int, default=2)
    parser.add_argument("--agg-k", type=int, default=12,
                        help="Number of clusters for agglomerative clustering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load task IDs for the source cluster
    inspection_path = Path(args.inspection)
    task_ids = parse_cluster_task_ids(inspection_path, args.source_cluster)
    if not task_ids:
        print(f"No tasks found for cluster {args.source_cluster}.")
        sys.exit(1)
    print(f"Source cluster {args.source_cluster}: {len(task_ids)} tasks")

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    all_ids = list(data["task_ids"])
    all_embs = data["embeddings"]
    all_descs = list(data["descriptions"])

    # Filter to source cluster tasks
    id_to_idx = {tid: i for i, tid in enumerate(all_ids)}
    idxs = [id_to_idx[tid] for tid in task_ids if tid in id_to_idx]
    embs = all_embs[idxs]
    descs = [all_descs[i] for i in idxs]
    task_ids = [all_ids[i] for i in idxs]
    print(f"  Embeddings shape: {embs.shape}")

    # UMAP on just these tasks
    print("Running UMAP...")
    n_neighbors = min(15, len(task_ids) - 1)
    reducer = UMAP(n_components=2, random_state=42,
                   n_neighbors=n_neighbors, min_dist=0.1)
    xy = reducer.fit_transform(embs)

    # --- Method A: HDBSCAN with looser params ---
    print(f"\nMethod A — HDBSCAN(min_cluster_size={args.hdbscan_min_cluster_size}, "
          f"min_samples={args.hdbscan_min_samples})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
    )
    hdb_labels = clusterer.fit_predict(xy)
    n_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = (hdb_labels == -1).sum()
    print(f"  {n_hdb} sub-clusters, {n_noise} noise points")
    for c in sorted(set(hdb_labels)):
        n = (hdb_labels == c).sum()
        label = f"Sub-cluster {c}" if c >= 0 else "Noise"
        print(f"    {label}: {n} tasks")

    plot_clusters(xy, hdb_labels, task_ids,
                  f"Cluster {args.source_cluster} re-clustered — HDBSCAN "
                  f"(min_cluster_size={args.hdbscan_min_cluster_size}, "
                  f"min_samples={args.hdbscan_min_samples})",
                  output_dir / "hdbscan_plot.png")
    write_inspection(task_ids, descs, hdb_labels,
                     output_dir / "hdbscan_inspection.txt")

    # --- Method B: Agglomerative (Ward) with fixed k ---
    print(f"\nMethod B — Agglomerative (Ward, k={args.agg_k})")
    Z = linkage(embs, method="ward")
    agg_labels = fcluster(Z, t=args.agg_k, criterion="maxclust") - 1  # 0-indexed
    for c in range(args.agg_k):
        n = (agg_labels == c).sum()
        print(f"    Sub-cluster {c}: {n} tasks")

    plot_clusters(xy, agg_labels, task_ids,
                  f"Cluster {args.source_cluster} re-clustered — Agglomerative "
                  f"Ward (k={args.agg_k})",
                  output_dir / "agglomerative_plot.png")
    write_inspection(task_ids, descs, agg_labels,
                     output_dir / "agglomerative_inspection.txt")

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
