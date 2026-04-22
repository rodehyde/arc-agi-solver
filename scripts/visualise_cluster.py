"""
visualise_cluster.py — Plot all train pairs for every task in a given cluster.

Usage:
    python scripts/visualise_cluster.py --cluster 1
    python scripts/visualise_cluster.py --cluster 0 --inspection results/cluster_inspection.txt
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loader import load_all_tasks
from src.visualise import plot_task


def parse_cluster_task_ids(inspection_path: Path, cluster: int) -> list[str]:
    """Extract task IDs for a given cluster from the inspection text file.

    Handles both 'Cluster N' (original) and 'Sub-cluster N' (re-clustered) formats.
    """
    text = inspection_path.read_text()
    # Try both label formats
    for label in (f"Cluster {cluster}", f"Sub-cluster {cluster}"):
        pattern = rf"{re.escape(label)} \(n=\d+\)\n={{{60}}}\n(.*?)(?:\n={{{60}}}|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            block = match.group(1)
            return re.findall(r"^\s{2}([0-9a-f]{8}):", block, re.MULTILINE)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=int, required=True, help="Cluster number to visualise")
    parser.add_argument("--inspection", default="results/cluster_inspection.txt")
    parser.add_argument("--output-dir", default="results/cluster_plots")
    args = parser.parse_args()

    inspection_path = Path(args.inspection)
    if not inspection_path.exists():
        print(f"Inspection file not found: {inspection_path}")
        print("Run embed_descriptions.py first.")
        sys.exit(1)

    task_ids = parse_cluster_task_ids(inspection_path, args.cluster)
    if not task_ids:
        print(f"No tasks found for cluster {args.cluster}.")
        sys.exit(1)

    print(f"Cluster {args.cluster}: {len(task_ids)} tasks — {task_ids}")

    # Load tasks
    all_tasks = load_all_tasks()
    task_map = {t["task_id"]: t for t in all_tasks}

    output_dir = Path(args.output_dir) / f"cluster_{args.cluster}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for tid in task_ids:
        task = task_map.get(tid)
        if task is None:
            print(f"  Task {tid} not found in training data, skipping.")
            continue
        fig = plot_task(task)
        out_path = output_dir / f"{tid}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"\nDone. {len(task_ids)} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
