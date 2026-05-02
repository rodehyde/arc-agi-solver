"""
compare_clusters.py — Compare old vs new cluster assignments for known task groups.

Pipeline:
  1. Embed new descriptions (data/descriptions_process.json) with MiniLM
  2. UMAP → 2D, then HDBSCAN (same params as original scene clustering)
  3. For each known group (C8, C3, pattern_restoration), report:
     - Where each task lands in the new clustering
     - What else is in that new cluster
     - Whether the two C8 outliers (1bfc4729, 7b6016b9) separate out

Usage:
    conda run -n arc-agi python scripts/compare_clusters.py
    conda run -n arc-agi python scripts/compare_clusters.py --min-cluster-size 4
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan

PROJECT_ROOT = Path(__file__).parent.parent

# ── Known groups ─────────────────────────────────────────────────────────────

GROUPS = {
    "C8_curated": [
        "09629e4f", "1190e5a7", "1e32b0e9", "272f95fa",
        "29623171", "54d9e175", "6773b310", "6d0160f0", "941d9a10",
    ],
    "C8_outliers": [
        "1bfc4729",   # line extension
        "7b6016b9",   # flood fill enclosed areas
    ],
    "C3_mirror": [
        "3af2c5a8", "49d1d64f", "4c4377d9", "62c24649", "67e8384a",
        "6d0aefbc", "6fa7a44f", "7fe24cdd", "8be77c9e", "8d5021e8",
        "a416b8f3", "c9e6f938",
    ],
    "pattern_restoration": [
        "3345333e", "3631a71a", "9ecd008a", "b8825c91",
        "0dfd9992", "29ec7d0e", "484b58aa", "73251a56", "c3f564a4",
    ],
}

ALL_KNOWN = {tid: grp for grp, tasks in GROUPS.items() for tid in tasks}


def load_old_clusters() -> dict[str, int]:
    """Parse results/cluster_inspection.txt → {task_id: cluster_id}."""
    import re
    path = PROJECT_ROOT / "results" / "cluster_inspection.txt"
    if not path.exists():
        return {}
    text = path.read_text()
    task_cluster = {}
    current_cluster = None
    for line in text.splitlines():
        m = re.match(r"^Cluster (\d+)", line)
        if m:
            current_cluster = int(m.group(1))
        m2 = re.match(r"^  ([0-9a-f]{8}):", line)
        if m2 and current_cluster is not None:
            task_cluster[m2.group(1)] = current_cluster
    return task_cluster


def embed_and_cluster(desc_path: Path, min_cluster_size: int, min_samples: int,
                      cache_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load descriptions, embed with MiniLM, UMAP+HDBSCAN. Returns (task_ids, xy, labels)."""
    descs_raw = json.loads(desc_path.read_text())
    task_ids = list(descs_raw.keys())
    texts = [descs_raw[t] if isinstance(descs_raw[t], str)
             else descs_raw[t].get("verified", "") for t in task_ids]

    # Embed (or load cache)
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        cached_ids = list(npz["task_ids"])
        if cached_ids == task_ids:
            print(f"  Loaded cached embeddings from {cache_path}")
            embeddings = npz["embeddings"]
        else:
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print("  Embedding with MiniLM-L6-v2 ...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        np.savez(cache_path, task_ids=task_ids, embeddings=embeddings)
        print(f"  Saved to {cache_path}")

    # UMAP → 2D
    print("  Running UMAP ...")
    n_neighbors = min(15, len(task_ids) - 1)
    reducer = UMAP(n_components=2, random_state=42,
                   n_neighbors=n_neighbors, min_dist=0.1)
    xy = reducer.fit_transform(embeddings)

    # HDBSCAN on 2D
    print(f"  Clustering (min_cluster_size={min_cluster_size}, min_samples={min_samples}) ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples)
    labels = clusterer.fit_predict(xy)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  → {n_clusters} clusters, {n_noise} noise points")

    return task_ids, xy, labels


def report_group(group_name: str, group_tasks: list[str],
                 new_labels: dict[str, int], new_cluster_members: dict[int, list[str]],
                 old_labels: dict[str, int], new_descs: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {group_name}  ({len(group_tasks)} tasks)")
    print(f"{'='*70}")

    # Where does each task land in new clustering?
    new_cluster_counts = Counter(new_labels.get(t, -1) for t in group_tasks)
    dominant_new = new_cluster_counts.most_common(1)[0][0]

    for tid in group_tasks:
        old_c = old_labels.get(tid, "?")
        new_c = new_labels.get(tid, -1)
        flag = " ★" if new_c == dominant_new else (" ✗noise" if new_c == -1 else f" →C{new_c}")
        print(f"  {tid}  old=C{old_c}  new=C{new_c}{flag}")

    # Summary of new cluster distribution
    print(f"\n  New cluster distribution: {dict(new_cluster_counts.most_common())}")
    purity = new_cluster_counts[dominant_new] / len(group_tasks) * 100
    print(f"  Dominant new cluster: C{dominant_new}  ({new_cluster_counts[dominant_new]}/{len(group_tasks)} = {purity:.0f}% purity)")

    # What else is in the dominant new cluster?
    if dominant_new != -1:
        members = new_cluster_members[dominant_new]
        outsiders = [t for t in members if t not in group_tasks]
        print(f"\n  C{dominant_new} full membership ({len(members)} tasks):")
        print(f"    In-group  ({len(members)-len(outsiders)}): {[t for t in members if t in group_tasks]}")
        print(f"    Outsiders ({len(outsiders)}): {outsiders}")

        # Print new descriptions for outsiders
        if outsiders and new_descs:
            print(f"\n  Outsider descriptions in C{dominant_new}:")
            for tid in outsiders[:5]:  # cap at 5
                desc = new_descs.get(tid, "")
                type_line = next((l for l in desc.splitlines() if l.startswith("TYPE:")), "")
                mech_line = next((l for l in desc.splitlines() if l.startswith("MECHANISM:")), "")
                known_grp = ALL_KNOWN.get(tid, "unknown")
                print(f"    {tid} [{known_grp}]")
                print(f"      {type_line}")
                print(f"      {mech_line[:120]}")


def write_inspection_file(new_labels: dict[str, int],
                          new_cluster_members: dict[int, list[str]],
                          new_descs: dict,
                          out_path: Path) -> None:
    """Write results/cluster_process_inspection.txt in the same format as cluster_inspection.txt."""
    cluster_ids = sorted(c for c in new_cluster_members if c != -1)
    n_clustered = sum(len(new_cluster_members[c]) for c in cluster_ids)
    n_noise = len(new_cluster_members.get(-1, []))
    lines = []
    lines.append(f"Cluster inspection — {len(new_labels)} tasks, {len(cluster_ids)} clusters")
    lines.append("")
    lines.append("")
    for cid in cluster_ids:
        members = new_cluster_members[cid]
        lines.append("=" * 60)
        lines.append(f"Cluster {cid} (n={len(members)})")
        lines.append("=" * 60)
        lines.append("")
        for tid in sorted(members):
            desc = new_descs.get(tid, "")
            lines.append(f"  {tid}:")
            for dline in desc.splitlines():
                lines.append(f"  {dline}")
            lines.append("")
    # Noise section
    noise_tasks = sorted(new_cluster_members.get(-1, []))
    if noise_tasks:
        lines.append("=" * 60)
        lines.append(f"Noise / unclustered (n={n_noise})")
        lines.append("=" * 60)
        lines.append("")
        for tid in noise_tasks:
            lines.append(f"  {tid}:")
            desc = new_descs.get(tid, "")
            for dline in desc.splitlines():
                lines.append(f"  {dline}")
            lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}  ({len(cluster_ids)} clusters, {n_noise} noise)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc-file",        default="data/descriptions_process.json")
    parser.add_argument("--embeddings-cache", default="data/embeddings_process.npz")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples",      type=int, default=2)
    args = parser.parse_args()

    desc_path = PROJECT_ROOT / args.desc_file
    cache_path = PROJECT_ROOT / args.embeddings_cache

    if not desc_path.exists():
        print(f"ERROR: {desc_path} not found. Run generate_descriptions.py first.")
        return

    new_descs = json.loads(desc_path.read_text())
    n_done = len(new_descs)
    print(f"Descriptions loaded: {n_done}/400")
    if n_done < 400:
        print(f"WARNING: only {n_done} descriptions — clustering may be incomplete.")

    # Old cluster assignments
    old_labels = load_old_clusters()
    print(f"Old cluster assignments loaded: {len(old_labels)} tasks")

    # Embed + cluster new descriptions
    print("\nEmbedding and clustering new descriptions ...")
    task_ids, xy, labels = embed_and_cluster(
        desc_path, args.min_cluster_size, args.min_samples, cache_path
    )

    new_labels = {tid: int(labels[i]) for i, tid in enumerate(task_ids)}
    new_cluster_members = defaultdict(list)
    for tid, c in new_labels.items():
        new_cluster_members[c].append(tid)

    # ── Save inspection file ──────────────────────────────────────────────────
    inspection_path = PROJECT_ROOT / "results" / "cluster_process_inspection.txt"
    write_inspection_file(new_labels, new_cluster_members, new_descs, inspection_path)

    # ── Report for each known group ──────────────────────────────────────────
    for group_name, group_tasks in GROUPS.items():
        report_group(group_name, group_tasks, new_labels, new_cluster_members,
                     old_labels, new_descs)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: old cluster → new cluster(s)")
    print(f"{'='*70}")
    for group_name, group_tasks in GROUPS.items():
        new_dist = Counter(new_labels.get(t, -1) for t in group_tasks)
        print(f"  {group_name:25s}: {dict(new_dist.most_common())}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
