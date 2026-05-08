"""
prepare_arc_finetune.py — Build nanoGPT fine-tuning dataset from ARC + LARC.

For each (task, verified-LARC-description) pair that fits within MAX_TOKENS:
  - Format as text: training examples → Rule → Test Input → Test Output
  - Tokenise with GPT-2 BPE (tiktoken)

Splits:
  - eval   : EVAL_PER_CATEGORY tasks per broad category (stratified, held out entirely)
  - val    : VAL_FRACTION of remaining tasks (by task, not by example)
  - train  : the rest

Outputs (written to data/finetune/):
  train.bin       — uint16 numpy array of token ids
  val.bin         — uint16 numpy array of token ids
  eval.jsonl      — one JSON object per eval example (human-readable)
  split_info.json — which tasks went where, token counts, category assignments

Usage:
  python scripts/prepare_arc_finetune.py
"""

import csv
import json
import random
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED              = 42
MAX_TOKENS        = 1024
EVAL_PER_CATEGORY = 4          # tasks held out per broad category
VAL_FRACTION      = 0.10       # fraction of remaining tasks used for val
EOT               = "<|endoftext|>"

PROJECT_ROOT             = Path(__file__).parent.parent
ARC_DIR                  = PROJECT_ROOT / "data" / "training"
LARC_DIR                 = PROJECT_ROOT / "data" / "larc"
OUT_DIR                  = PROJECT_ROOT / "data" / "finetune"
CLAUDE_DESCRIPTIONS_FILE = PROJECT_ROOT / "data" / "claude_descriptions.json"

# Map Claude TYPE labels → broad categories
CLAUDE_TYPE_MAP = {
    "geometric": "GEOMETRIC",
    "fill":      "FILL",
    "count":     "CLASSIFY_COUNT",
    "extend":    "EXTEND",
    "crop":      "CROP_EXTRACT",
    "other":     "OTHER",
}

# ---------------------------------------------------------------------------
# Broad category keyword matching (same logic as descriptions_process.json)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "GEOMETRIC": [
        "rotat", "reflect", "mirror", "flip", "copy-and-place", "translat",
        "scale", "tile", "fold", "symmetr", "shear", "stamp mirrored",
    ],
    "FILL": [
        "flood", "fill", "inpaint", "gap fill", "border fill", "reconstruct",
        "enclosed", "interior", "patch",
    ],
    "CLASSIFY_COUNT": [
        "count", "classif", "rank", "xor", " or ", " and ", "recolour",
        "recolor", "majority", "lookup", "substitut", "modular",
    ],
    "CROP_EXTRACT": [
        "crop", "extract", "select", "downsample", "filter", "isolat",
        "bounding box", "unique-colour", "unique-color",
    ],
    "EXTEND": [
        "extend", "propagat", "diagonal", "extrapolat", "ray", "stripe",
        "periodic", "align", "gravity", "project",
    ],
}


def assign_category(type_text: str) -> str:
    """Return the first matching broad category, or 'OTHER'."""
    lower = type_text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return cat
    return "OTHER"


# ---------------------------------------------------------------------------
# Grid / text formatting
# ---------------------------------------------------------------------------

def format_grid(grid: list) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_example(task: dict, rule: str) -> str:
    parts = []
    for i, pair in enumerate(task["train"]):
        parts.append(f"Example {i+1} Input:\n{format_grid(pair['input'])}")
        parts.append(f"Example {i+1} Output:\n{format_grid(pair['output'])}")
    test = task["test"][0]
    parts.append(f"Test Input:\n{format_grid(test['input'])}")
    parts.append(f"Rule: {rule}")
    parts.append(f"Test Output:\n{format_grid(test['output'])}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Load LARC (verified descriptions only)
# ---------------------------------------------------------------------------

def load_larc() -> dict[str, list[str]]:
    """Return {task_id: [verified rule descriptions]}."""
    # description_id -> description_output text
    desc_text: dict[str, str] = {}
    with open(LARC_DIR / "larc_description.csv") as f:
        for row in csv.DictReader(f):
            if row["is_verified"] == "True":
                rule = row["description_output"].strip()
                if rule:
                    desc_text[row["description_id"]] = rule

    # task row index -> task_id (strip .json suffix)
    task_index: dict[str, str] = {}
    with open(LARC_DIR / "larc_task.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            task_index[str(i)] = row["task_name"].replace(".json", "")

    # join: task_id -> list of rule strings
    task_rules: dict[str, list[str]] = defaultdict(list)
    with open(LARC_DIR / "larc_join.csv") as f:
        for row in csv.DictReader(f):
            tid = task_index.get(row["task_id"])
            did = row["description_id"]
            if tid and did in desc_text:
                task_rules[tid].append(desc_text[did])

    # Deduplicate rules per task
    return {tid: list(dict.fromkeys(rules)) for tid, rules in task_rules.items()}


# ---------------------------------------------------------------------------
# Load ARC tasks
# ---------------------------------------------------------------------------

def load_arc_tasks() -> dict[str, dict]:
    tasks = {}
    for p in ARC_DIR.glob("*.json"):
        t = json.loads(p.read_text())
        if "output" in t["test"][0]:   # keep only tasks with known test output
            tasks[p.stem] = t
    return tasks


# ---------------------------------------------------------------------------
# Load broad categories from descriptions_process.json
# ---------------------------------------------------------------------------

def load_claude_descriptions() -> dict[str, str]:
    """Return {task_id: full_description_text} from Claude-generated descriptions."""
    if not CLAUDE_DESCRIPTIONS_FILE.exists():
        return {}
    return json.loads(CLAUDE_DESCRIPTIONS_FILE.read_text())


def load_categories(claude_descs: dict[str, str] | None = None) -> dict[str, str]:
    """Return {task_id: broad_category}.

    Claude TYPE labels take priority; falls back to keyword matching on
    descriptions_process.json for any tasks not covered by Claude.
    """
    cats: dict[str, str] = {}

    # Claude descriptions carry explicit TYPE labels — most reliable source
    if claude_descs:
        type_re = re.compile(r"^TYPE:\s*(\w+)", re.IGNORECASE | re.MULTILINE)
        for tid, desc in claude_descs.items():
            m = type_re.search(desc)
            if m:
                label = m.group(1).strip().lower()
                cats[tid] = CLAUDE_TYPE_MAP.get(label, "OTHER")

    # Fallback: keyword matching on older descriptions_process.json
    path = PROJECT_ROOT / "data" / "descriptions_process.json"
    if path.exists():
        raw = json.loads(path.read_text())
        type_re2 = re.compile(r"TYPE:\s*([^\n]+)", re.IGNORECASE)
        for tid, text in raw.items():
            if tid not in cats:
                m = type_re2.search(text)
                cats[tid] = assign_category(m.group(1).strip()) if m else "OTHER"

    return cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token   # 50256

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    arc_tasks   = load_arc_tasks()
    larc_rules  = load_larc()
    claude_descs = load_claude_descriptions()
    categories  = load_categories(claude_descs)

    print(f"  ARC tasks with test output  : {len(arc_tasks)}")
    print(f"  Tasks with LARC descriptions: {len(larc_rules)}")
    print(f"  Tasks with Claude descriptions: {len(claude_descs)}")

    # ------------------------------------------------------------------
    # Build candidate pool: (task_id, rule, token_count, text)
    # ------------------------------------------------------------------
    pool: dict[str, list[dict]] = defaultdict(list)   # task_id -> examples

    # LARC descriptions
    for tid, rules in larc_rules.items():
        if tid not in arc_tasks:
            continue
        task = arc_tasks[tid]
        for rule in rules:
            text = format_example(task, rule)
            tokens = enc.encode_ordinary(text)
            if len(tokens) <= MAX_TOKENS:
                pool[tid].append({
                    "task_id": tid,
                    "rule":    rule,
                    "tokens":  len(tokens),
                    "text":    text,
                    "source":  "larc",
                })

    # Claude descriptions (TYPE / RULE / STEPS / RELATIONSHIP block)
    for tid, desc in claude_descs.items():
        if tid not in arc_tasks:
            continue
        task = arc_tasks[tid]
        text = format_example(task, desc)
        tokens = enc.encode_ordinary(text)
        if len(tokens) <= MAX_TOKENS:
            pool[tid].append({
                "task_id": tid,
                "rule":    desc,
                "tokens":  len(tokens),
                "text":    text,
                "source":  "claude",
            })

    eligible_tasks = list(pool.keys())
    n_larc   = sum(1 for v in pool.values() for ex in v if ex.get("source") == "larc")
    n_claude = sum(1 for v in pool.values() for ex in v if ex.get("source") == "claude")
    print(f"  Eligible tasks (≤{MAX_TOKENS} tokens): {len(eligible_tasks)}")
    print(f"  Examples — LARC: {n_larc}  Claude: {n_claude}  Total: {n_larc + n_claude}")

    # ------------------------------------------------------------------
    # Stratified eval split: EVAL_PER_CATEGORY tasks per category
    # ------------------------------------------------------------------
    # Group eligible tasks by category
    by_cat: dict[str, list[str]] = defaultdict(list)
    for tid in eligible_tasks:
        cat = categories.get(tid, "OTHER")
        by_cat[cat].append(tid)

    print("\nCategory breakdown of eligible tasks:")
    for cat, tids in sorted(by_cat.items()):
        print(f"  {cat:20s}: {len(tids):3d} tasks")

    eval_tasks: set[str] = set()
    for cat, tids in by_cat.items():
        rng.shuffle(tids)
        n = min(EVAL_PER_CATEGORY, len(tids))
        eval_tasks.update(tids[:n])

    print(f"\nEval tasks selected (stratified): {len(eval_tasks)}")

    # ------------------------------------------------------------------
    # Train / val split on remaining tasks
    # ------------------------------------------------------------------
    remaining = [tid for tid in eligible_tasks if tid not in eval_tasks]
    rng.shuffle(remaining)
    n_val = max(1, int(len(remaining) * VAL_FRACTION))
    val_tasks  = set(remaining[:n_val])
    train_tasks = set(remaining[n_val:])

    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks  : {len(val_tasks)}")

    # ------------------------------------------------------------------
    # Tokenise and save train.bin / val.bin
    # ------------------------------------------------------------------
    def tokenise_split(task_set: set[str]) -> np.ndarray:
        all_tokens = []
        for tid in sorted(task_set):
            for ex in pool[tid]:
                toks = enc.encode_ordinary(ex["text"])
                all_tokens.extend(toks)
                all_tokens.append(eot_token)
        return np.array(all_tokens, dtype=np.uint16)

    print("\nTokenising splits...")
    train_arr = tokenise_split(train_tasks)
    val_arr   = tokenise_split(val_tasks)

    (OUT_DIR / "train.bin").write_bytes(train_arr.tobytes())
    (OUT_DIR / "val.bin").write_bytes(val_arr.tobytes())
    print(f"  train.bin : {len(train_arr):,} tokens")
    print(f"  val.bin   : {len(val_arr):,} tokens")

    # ------------------------------------------------------------------
    # Save eval.jsonl (human-readable, one line per example)
    # ------------------------------------------------------------------
    eval_lines = []
    for tid in sorted(eval_tasks):
        task = arc_tasks[tid]
        cat  = categories.get(tid, "OTHER")
        for ex in pool[tid]:
            eval_lines.append(json.dumps({
                "task_id":  tid,
                "category": cat,
                "tokens":   ex["tokens"],
                "rule":     ex["rule"],
                "text":     ex["text"],
            }))

    (OUT_DIR / "eval.jsonl").write_text("\n".join(eval_lines) + "\n")
    print(f"  eval.jsonl: {len(eval_lines)} examples across {len(eval_tasks)} tasks")

    # ------------------------------------------------------------------
    # Save split_info.json for reproducibility
    # ------------------------------------------------------------------
    split_info = {
        "seed":              SEED,
        "max_tokens":        MAX_TOKENS,
        "eval_per_category": EVAL_PER_CATEGORY,
        "val_fraction":      VAL_FRACTION,
        "train_tasks":       sorted(train_tasks),
        "val_tasks":         sorted(val_tasks),
        "eval_tasks":        sorted(eval_tasks),
        "category_of_eval":  {tid: categories.get(tid, "OTHER") for tid in sorted(eval_tasks)},
        "train_tokens":      int(len(train_arr)),
        "val_tokens":        int(len(val_arr)),
    }
    (OUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2))
    print("  split_info.json saved")
    print("\nDone.")


if __name__ == "__main__":
    main()
