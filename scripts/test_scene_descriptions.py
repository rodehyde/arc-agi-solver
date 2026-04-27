"""
test_scene_descriptions.py — Test scene-first task descriptions against the
current structured-vocabulary approach.

Runs on a small set of hand-picked tasks and prints both old and new
descriptions side by side so we can evaluate quality.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/test_scene_descriptions.py
    python scripts/test_scene_descriptions.py --model claude-haiku-4-5-20251001
    python scripts/test_scene_descriptions.py --tasks 2dd70a9a 272f95fa
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loader import load_all_tasks

PROJECT_ROOT = Path(__file__).parent.parent

# ── Default test tasks ────────────────────────────────────────────────────────
# Chosen because we already have ground-truth human descriptions for comparison:
#   2dd70a9a  — guided navigation (path steered by grey bumpers)
#   272f95fa  — compartment fill (fixed colour map onto partitioned grid)
#   29c11459  — EXTEND_LINE disguised as EXTEND_RECT (simple horizontal fill)
#   00d62c1b  — genuine EXTEND_RECT (draw rectangle border around markers)
#   3c9b0459  — STRUCTURE_UNCHANGED (pure colour mapping, rule-based solvable)

DEFAULT_TASKS = [
    "2dd70a9a",  # guided navigation
    "272f95fa",  # compartment fill
    "29c11459",  # degenerate EXTEND_LINE in EXTEND_RECT
    "00d62c1b",  # genuine EXTEND_RECT
    "3c9b0459",  # STRUCTURE_UNCHANGED / colour mapping
]

# ── Old prompt (current approach) ────────────────────────────────────────────

OLD_SYSTEM = """\
You are an expert at analysing abstract visual puzzles. You will be shown training \
examples from an ARC task. Each example has an input grid and an output grid. \
Colours are represented as integers 0–9.

Output format (use exactly these five labelled lines, nothing else):

TASK TYPE: <whole-grid-transform | crop-extract | locate-then-extract | multi-stage-rule>
SIZE CHANGE: <same | shrink | other-grow>
STEPS: <1 | 2 | 3+>
LOCATING NEEDED: <yes | no>
OPERATION: <one sentence using only controlled verbs: flip-h, flip-v, rotate-90cw,
rotate-180, crop <region>, locate-mirror-position-then-crop,
assembly-rule: <brief description>>

No reasoning, no analysis, no extra text."""

OLD_USER = """\
Task ID: {task_id}
Training examples ({n_pairs} pairs):

{pairs_text}

Output ONLY the five template lines:

TASK TYPE:
SIZE CHANGE:
STEPS:
LOCATING NEEDED:
OPERATION: """

# ── New scene-first prompt ────────────────────────────────────────────────────

NEW_SYSTEM = """\
You are an expert at analysing abstract visual puzzles (ARC tasks).
Each task shows input/output grid pairs. Colours are integers 0–9; 0 is background.

Your goal: write a description that captures the MECHANISM — the rule that explains
why the output looks the way it does. Another AI reading your description should be
able to solve a new example without seeing any grids.

Work through this exact sequence before writing your answer:

1. SCENE: What does each input look like as a scene? Name the role of each colour
   (e.g. "a sparse scatter of grey cells acting as obstacles", "two small coloured
   dominoes embedded in noise", "a grid of dividing lines carving out rooms").
   Do NOT describe the output yet.

2. INVARIANTS: What is the same across ALL input examples? (Which colours appear?
   How many objects? Fixed spatial structure?)

3. VARIABLES: What differs between examples? (Sizes, positions, which colours used?)

4. MECHANISM: What rule maps each input to its output? Describe it in terms of the
   roles you named — not what cells were added, but WHY those cells were added.
   Capture the intent, not the geometry.

5. TYPE: Invent 1–4 words that name the KIND of task (e.g. "guided navigation",
   "compartment fill", "colour remapping", "border completion"). Do not use
   the words "grid", "task", or "ARC".

Output format — use exactly these five labelled lines:
SCENE: <1-2 sentences>
INVARIANTS: <1-2 sentences>
VARIABLES: <1 sentence>
MECHANISM: <2-3 sentences describing the rule functionally>
TYPE: <1-4 words>"""

NEW_USER = """\
Task ID: {task_id}
Training examples ({n_pairs} pairs):

{pairs_text}

Now work through the five steps and give your five labelled lines."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def format_grid(grid):
    return "\n".join("  " + " ".join(str(c) for c in row) for row in grid)


def format_pairs(task, max_pairs=3):
    parts = []
    for i, p in enumerate(task["train"][:max_pairs]):
        parts.append(f"Pair {i + 1}:")
        parts.append(f"  Input  ({len(p['input'])}r × {len(p['input'][0])}c):")
        parts.append(format_grid(p["input"]))
        parts.append(f"  Output ({len(p['output'])}r × {len(p['output'][0])}c):")
        parts.append(format_grid(p["output"]))
    return "\n".join(parts)


def call(client, system, user, model, max_tokens):
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        timeout=60,
    )
    return msg.content[0].text.strip()


def run_task(client, task, model):
    pairs_text = format_pairs(task)
    tid = task["task_id"]
    n = min(3, len(task["train"]))

    old = call(client, OLD_SYSTEM,
               OLD_USER.format(task_id=tid, n_pairs=n, pairs_text=pairs_text),
               model, max_tokens=120)

    new = call(client, NEW_SYSTEM,
               NEW_USER.format(task_id=tid, n_pairs=n, pairs_text=pairs_text),
               model, max_tokens=400)

    return old, new


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model (default: claude-sonnet-4-6 for quality test)")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    client = anthropic.Anthropic()

    all_tasks = {t["task_id"]: t for t in load_all_tasks()}
    missing = [tid for tid in args.tasks if tid not in all_tasks]
    if missing:
        print(f"WARNING: tasks not found: {missing}")

    task_list = [all_tasks[tid] for tid in args.tasks if tid in all_tasks]
    print(f"Model: {args.model}")
    print(f"Tasks: {[t['task_id'] for t in task_list]}\n")
    print("=" * 70)

    results = {}
    for task in task_list:
        tid = task["task_id"]
        print(f"\n{'═' * 70}")
        print(f"  TASK: {tid}  ({len(task['train'])} training pairs)")
        print(f"{'═' * 70}")

        try:
            old, new = run_task(client, task, args.model)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print("\n── OLD (structured vocabulary) ──────────────────────────────")
        for line in old.splitlines():
            print(f"  {line}")

        print("\n── NEW (scene-first) ────────────────────────────────────────")
        for line in new.splitlines():
            print(f"  {line}")

        results[tid] = {"old": old, "new": new}
        time.sleep(args.delay)

    # Save results for later comparison
    out_path = PROJECT_ROOT / "results" / "scene_description_test.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
