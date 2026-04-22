"""
embed_descriptions.py — Embed task descriptions and visualise clusters.

Reads data/descriptions_training.json (produced by generate_descriptions.py).
Falls back to data/descriptions_sample.json if the full file doesn't exist yet.

Usage:
    python scripts/embed_descriptions.py
    python scripts/embed_descriptions.py --input data/descriptions_training.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


def load_descriptions(path: str) -> tuple[list[str], list[str]]:
    """Return (task_ids, descriptions). Handles both flat and profiled formats."""
    with open(path) as f:
        data = json.load(f)
    task_ids, descriptions = [], []
    for tid, value in data.items():
        task_ids.append(tid)
        if isinstance(value, dict):
            # refined format uses "verified"; original format uses "description"
            descriptions.append(value.get("verified") or value.get("description") or "")
        else:
            descriptions.append(value)
    return task_ids, descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="Path to descriptions JSON. Auto-detects training vs sample.",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--output", default="results/description_clusters.png")
    parser.add_argument(
        "--save-embeddings", default="data/embeddings_training.npz",
        help="Save embeddings + task_ids for later retrieval use."
    )
    args = parser.parse_args()

    # Choose input file
    if args.input:
        input_path = args.input
    elif Path("data/descriptions_training.json").exists():
        input_path = "data/descriptions_training.json"
    else:
        input_path = "data/descriptions_sample.json"
        print("Using sample descriptions (run generate_descriptions.py for full set).")

    print(f"Loading descriptions from {input_path}")
    task_ids, descriptions = load_descriptions(input_path)
    print(f"  {len(task_ids)} tasks loaded.")

    # Embed
    print(f"Embedding with '{args.model}'...")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(descriptions, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # Save embeddings for retrieval use
    os.makedirs(os.path.dirname(args.save_embeddings) or ".", exist_ok=True)
    np.savez(args.save_embeddings, task_ids=task_ids, embeddings=embeddings,
             descriptions=descriptions)
    print(f"  Embeddings saved to {args.save_embeddings}")

    # UMAP
    n_neighbors = min(15, len(task_ids) - 1)
    reducer = UMAP(n_components=2, random_state=42,
                   n_neighbors=n_neighbors, min_dist=0.1)
    xy = reducer.fit_transform(embeddings)

    # Cluster colouring — run HDBSCAN on 2D UMAP coords (more reliable than high-dim)
    if len(task_ids) >= 20:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
            labels = clusterer.fit_predict(xy)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            print(f"  HDBSCAN found {n_clusters} clusters ({n_noise} noise points).")
            cmap = plt.cm.tab20
            point_colours = [cmap(l / max(n_clusters, 1)) if l >= 0
                             else (0.7, 0.7, 0.7, 1.0) for l in labels]
        except ImportError:
            labels = np.zeros(len(task_ids), dtype=int)
            n_clusters = 0
            point_colours = ["steelblue"] * len(task_ids)
    else:
        labels = np.zeros(len(task_ids), dtype=int)
        n_clusters = 0
        point_colours = ["steelblue"] * len(task_ids)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, tid in enumerate(task_ids):
        ax.scatter(xy[i, 0], xy[i, 1], color=point_colours[i], s=60, zorder=3)
        if len(task_ids) <= 50:
            ax.annotate(tid, (xy[i, 0], xy[i, 1]), fontsize=6, ha="left",
                        va="bottom", xytext=(3, 3), textcoords="offset points")

    # Legend: one entry per cluster + noise
    if n_clusters > 0:
        cmap = plt.cm.tab20
        handles = [mpatches.Patch(color=cmap(c / max(n_clusters, 1)),
                                  label=f"Cluster {c} (n={(labels == c).sum()})")
                   for c in range(n_clusters)]
        handles.append(mpatches.Patch(color=(0.7, 0.7, 0.7),
                                      label=f"Noise (n={(labels == -1).sum()})"))
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  ncol=max(1, n_clusters // 15), framealpha=0.7)

    ax.set_title(f"Task description embeddings — {len(task_ids)} tasks "
                 f"({n_clusters} clusters, UMAP 2D)")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"  Plot saved to {args.output}")

    # Small cluster inspection — print and save
    if n_clusters > 0:
        sim = cosine_similarity(embeddings)
        np.fill_diagonal(sim, -1)

        report_lines = []
        report_lines.append(f"Cluster inspection — {len(task_ids)} tasks, {n_clusters} clusters\n")

        for c in range(n_clusters):
            idxs = [i for i in range(len(task_ids)) if labels[i] == c]
            members = [(task_ids[i], descriptions[i]) for i in idxs]
            report_lines.append(f"\n{'='*60}")
            report_lines.append(f"Cluster {c} (n={len(members)})")
            report_lines.append(f"{'='*60}")
            for tid, desc in members:
                report_lines.append(f"\n  {tid}:")
                report_lines.append(f"  {desc}")

        noise_idxs = [i for i in range(len(task_ids)) if labels[i] == -1]
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"Noise (n={len(noise_idxs)})")
        report_lines.append(f"{'='*60}")
        for i in noise_idxs:
            report_lines.append(f"\n  {task_ids[i]}:")
            report_lines.append(f"  {descriptions[i]}")

        report_lines.append(f"\n\n{'='*60}")
        report_lines.append("Nearest neighbours (cosine similarity) — first 20 tasks")
        report_lines.append(f"{'='*60}")
        for i, tid in enumerate(task_ids[:20]):
            nn_idx = int(np.argmax(sim[i]))
            report_lines.append(f"\n  {tid}  →  {task_ids[nn_idx]}  sim={sim[i, nn_idx]:.3f}")
            report_lines.append(f"    A: {descriptions[i]}")
            report_lines.append(f"    B: {descriptions[nn_idx]}")

        report_path = Path("results/cluster_inspection.txt")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text("\n".join(report_lines))
        print(f"  Cluster inspection saved to {report_path}")
        print("\n--- Small cluster inspection ---")
        for line in report_lines:
            print(line)


if __name__ == "__main__":
    main()
