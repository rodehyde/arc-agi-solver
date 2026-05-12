"""
download_barc.py — Download the BARC synthetic ARC dataset from HuggingFace.

Saves each problem as a JSON file in data/barc/<id>.json using standard ARC format:
  { "train": [ {"input": [[...]], "output": [[...]]}, ... ] }

Only train pairs are saved (test pairs from BARC are not used — we generate
our own leave-one-out test splits during training).

Resume-safe: already-converted files are skipped, so the script can be
interrupted and restarted without losing progress.

Usage:
    python scripts/download_barc.py
    python scripts/download_barc.py --limit 5000   # save first 5000 tasks
    python scripts/download_barc.py --split train   # HuggingFace split (default: train)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "barc"


def _to_grid(obj) -> list[list[int]] | None:
    """Convert a grid from any common representation to list[list[int]]."""
    if obj is None:
        return None
    if isinstance(obj, list):
        if not obj:
            return None
        # Already list[list[int]] or list[list[float]]
        if isinstance(obj[0], list):
            return [[int(c) for c in row] for row in obj]
        # Flat list — not expected but guard anyway
        return None
    # Some datasets store grids as dicts with a 'grid' key
    if isinstance(obj, dict) and "grid" in obj:
        return _to_grid(obj["grid"])
    return None


def _extract_pairs(example: dict) -> list[dict] | None:
    """Return a list of {input, output} dicts from a BARC example, or None."""
    # ── Format 1: ARC-standard  {'train': [{'input': ..., 'output': ...}, ...]} ──
    if "train" in example and isinstance(example["train"], list):
        pairs = []
        for p in example["train"]:
            if not isinstance(p, dict):
                break
            inp = _to_grid(p.get("input"))
            out = _to_grid(p.get("output"))
            if inp is not None and out is not None:
                pairs.append({"input": inp, "output": out})
        if len(pairs) >= 2:
            return pairs

    # ── Format 2: split arrays  {'train_inputs': [...], 'train_outputs': [...]} ──
    if "train_inputs" in example and "train_outputs" in example:
        inputs  = example["train_inputs"]
        outputs = example["train_outputs"]
        if isinstance(inputs, list) and isinstance(outputs, list):
            pairs = []
            for inp_raw, out_raw in zip(inputs, outputs):
                inp = _to_grid(inp_raw)
                out = _to_grid(out_raw)
                if inp is not None and out is not None:
                    pairs.append({"input": inp, "output": out})
            if len(pairs) >= 2:
                return pairs

    # ── Format 3: nested under a 'task' key ──────────────────────────────────
    if "task" in example:
        return _extract_pairs(example["task"])

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",  type=int, default=None,
                    help="Cap the number of tasks saved (default: all)")
    ap.add_argument("--split",  default="train",
                    help="HuggingFace dataset split to load (default: train)")
    ap.add_argument("--repo",   default="xu3kev/BARC",
                    help="HuggingFace dataset repo ID (default: xu3kev/BARC)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help=f"Output directory (default: {OUT_DIR})")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("  Run:  pip install datasets")
        sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in out.glob("*.json")}
    print(f"Loading {args.repo} (split={args.split}) ...")
    ds = load_dataset(args.repo, split=args.split, trust_remote_code=True)
    print(f"  {len(ds)} examples in dataset.")

    # Show keys of first example to help diagnose format issues
    first = ds[0]
    print(f"  Example keys: {list(first.keys())}")

    saved = skipped = failed = 0
    limit = args.limit or len(ds)

    for i, example in enumerate(ds):
        if saved >= limit:
            break

        task_id = f"{i:06d}"
        if task_id in existing:
            skipped += 1
            continue

        pairs = _extract_pairs(example)
        if pairs is None:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: could not extract pairs from example {i} "
                      f"(keys: {list(example.keys())})")
            continue

        # Filter: grids must be 1×1 to 30×30 and all values 0–9
        ok = True
        for p in pairs:
            for g in (p["input"], p["output"]):
                if not g or len(g) > 30 or any(len(r) > 30 or
                        any(c < 0 or c > 9 for c in r) for r in g):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            failed += 1
            continue

        (out / f"{task_id}.json").write_text(json.dumps({"train": pairs}))
        saved += 1

        if saved % 1000 == 0:
            print(f"  {saved} tasks saved  (skipped={skipped}, failed={failed})")

    print(f"\nDone.  saved={saved}  skipped={skipped}  failed={failed}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
