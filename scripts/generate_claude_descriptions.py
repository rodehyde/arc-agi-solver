"""
generate_claude_descriptions.py — Use Claude to generate structured rule
descriptions for ARC training tasks.

For each task, Claude produces a single description with four sections:
  TYPE         — task category (geometric / fill / count / extend / crop / other)
  RULE         — one-sentence summary of the transformation
  STEPS        — numbered procedural steps precise enough to reconstruct the output
  RELATIONSHIP — what stays fixed, what changes, and any exceptions

Progress is saved after every task so the script can be safely interrupted
and resumed from where it left off.

Output: data/claude_descriptions.json
  {task_id: description_string, ...}

Usage:
  python scripts/generate_claude_descriptions.py            # all 400 tasks
  python scripts/generate_claude_descriptions.py --limit 5  # first 5 (testing)
  python scripts/generate_claude_descriptions.py --task-id 007bbfb7
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent.parent
ARC_DIR      = PROJECT_ROOT / "data" / "training"
OUT_FILE     = PROJECT_ROOT / "data" / "claude_descriptions.json"

MODEL       = "claude-sonnet-4-6"
CALL_DELAY  = 0.3   # seconds between successful calls (stay under rate limit)
RETRY_DELAY = 2.0   # base seconds on rate-limit error (doubles each retry)

SYSTEM_PROMPT = """\
You are analysing ARC (Abstraction and Reasoning Corpus) puzzles.
Each puzzle has a hidden transformation rule that maps input grids to output grids.

Your descriptions will be used to train a small language model to predict output
grids. They must be precise enough that someone who cannot see the training
examples could reconstruct the correct output from your description plus only
the test input grid.

Grids are represented as rows of space-separated digit values (0–9).

Rules of thumb for useful descriptions:
- Name colours by their digit value AND a plain-English name where possible
  (e.g. "colour 3 (green)", "colour 0 (background/black)").
- Give exact directions (up, down, left, right, diagonal, clockwise).
- Give exact counts or sizes where they matter.
- If there is a conditional ("only if the cell is non-zero"), state it.
- Vague descriptions like "mirror it" or "apply a pattern" are not useful.\
"""

USER_TEMPLATE = """\
{task_text}

Study the examples above and identify the transformation rule.

Respond in exactly this format — four labelled sections, nothing else:

TYPE: one of: geometric / fill / count / extend / crop / other

RULE: one sentence that captures the complete transformation.

STEPS:
1. ...
2. ...
(Number every step. Be specific about colour digit values, row/column positions,
directions, and sizes. Include any conditions or special cases.)

RELATIONSHIP: What stays the same between input and output? What changes?
Are there any exceptions or boundary conditions?\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_grid(grid: list) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_task(task: dict) -> str:
    parts = []
    for i, pair in enumerate(task["train"]):
        parts.append(f"Example {i+1} Input:\n{format_grid(pair['input'])}")
        parts.append(f"Example {i+1} Output:\n{format_grid(pair['output'])}")
    return "\n\n".join(parts)


def describe_task(client: anthropic.Anthropic, task: dict, retries: int = 3) -> str:
    prompt = USER_TEMPLATE.format(task_text=format_task(task))
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f" [rate limit — waiting {wait:.0f}s]", end="", flush=True)
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(RETRY_DELAY)
    raise RuntimeError("All retries exhausted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Claude rule descriptions for ARC training tasks."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N tasks (useful for testing)",
    )
    parser.add_argument(
        "--task-id", type=str, default=None,
        help="Process a single task by ID and print the result",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL,
        help=f"Claude model to use (default: {MODEL})",
    )
    parser.add_argument(
        "--out-file", type=str, default=None,
        help="Override output path (default: data/claude_descriptions.json). "
             "Useful in Colab to save directly to Drive.",
    )
    args = parser.parse_args()

    out_file = Path(args.out_file) if args.out_file else OUT_FILE

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

    # Load saved progress
    results: dict[str, str] = {}
    if out_file.exists():
        results = json.loads(out_file.read_text())
        print(f"Resuming: {len(results)} tasks already saved.")

    # Load ARC tasks
    all_tasks: dict[str, dict] = {}
    for p in sorted(ARC_DIR.glob("*.json")):
        all_tasks[p.stem] = json.loads(p.read_text())

    # Select tasks to process
    if args.task_id:
        todo = [args.task_id] if args.task_id in all_tasks else []
        if not todo:
            print(f"Task {args.task_id!r} not found in {ARC_DIR}")
            return
    else:
        todo = [tid for tid in all_tasks if tid not in results]

    if args.limit:
        todo = todo[: args.limit]

    print(f"Tasks to process: {len(todo)}")

    for i, tid in enumerate(todo):
        print(f"  [{i+1:3d}/{len(todo)}] {tid} ...", end=" ", flush=True)
        try:
            desc = describe_task(client, all_tasks[tid])
            results[tid] = desc
            out_file.write_text(json.dumps(results, indent=2))
            print("ok")
            if args.task_id:          # single-task mode: print in full
                print("\n" + "=" * 60)
                print(desc)
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(todo) - 1:        # no sleep after last task
            time.sleep(CALL_DELAY)

    print(f"\nDone. {len(results)}/{len(all_tasks)} tasks saved → {out_file}")


if __name__ == "__main__":
    main()
