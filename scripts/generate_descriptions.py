"""
generate_descriptions.py — Use Claude to generate scene-first natural language
descriptions of ARC training tasks.

Uses a scene-first prompting approach: read the input as a scene before touching
the output, name the role of each colour, question apparent randomness, identify
invariants vs variables, then describe the mechanism functionally.

Descriptions are saved incrementally to data/descriptions_scene.json so the
script can be safely interrupted and restarted.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_descriptions.py
    python scripts/generate_descriptions.py --limit 10   # test on first 10 tasks

Options:
    --limit N       Only process the first N tasks (default: all)
    --delay SECS    Seconds to wait between API calls (default: 0.5)
    --model NAME    Claude model to use (default: claude-haiku-4-5-20251001)
    --force         Re-generate descriptions even if already present
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
OUTPUT_PATH = PROJECT_ROOT / "data" / "descriptions_scene.json"

SYSTEM_PROMPT = """\
You are an expert at analysing abstract visual puzzles (ARC tasks).
Each task shows input/output grid pairs. Colours are integers 0–9; 0 is background.

Your goal: write a description that captures the MECHANISM — the rule that explains
why the output looks the way it does. Another AI reading your description should be
able to solve a new example of this task without seeing any grids.

Work through this exact sequence:

1. SCENE: What does each input look like as a scene? Name the role of each colour
   (e.g. "a sparse scatter of grey cells acting as obstacles", "two small coloured
   dominoes embedded in noise", "a grid of dividing lines carving out rooms").
   Then look at whatever seems random or meaningless — scattered cells, irregular
   patterns, apparent noise — and ask whether it might be functional rather than
   decorative. Do NOT describe the output yet.

2. INVARIANTS: What is the same across ALL input examples? (Which colours always
   appear? How many objects? Any fixed spatial structure?)

3. VARIABLES: What differs between examples? (Sizes, positions, which colours used?)

4. MECHANISM: What rule maps each input to its output? Describe it in terms of the
   roles you named — not what cells were added, but WHY those cells were added.
   Capture the intent, not the geometry.

5. TYPE: Invent 1–4 words naming the KIND of task (e.g. "guided navigation",
   "compartment fill", "colour remapping", "border completion"). Be specific —
   prefer "enclosed region flood fill" over "fill task".

Output exactly these five labelled lines and nothing else:
SCENE: <1-2 sentences>
INVARIANTS: <1-2 sentences>
VARIABLES: <1 sentence>
MECHANISM: <2-3 sentences>
TYPE: <1-4 words>"""

USER_TEMPLATE = """\
Task ID: {task_id}

Training examples ({n_pairs} pairs):

{pairs_text}

Output the five labelled lines (SCENE / INVARIANTS / VARIABLES / MECHANISM / TYPE):"""


def format_grid(grid: list[list[int]]) -> str:
    return "\n".join("  " + " ".join(str(c) for c in row) for row in grid)


def format_pairs(task: dict, max_pairs: int = 3) -> str:
    parts = []
    for i, p in enumerate(task["train"][:max_pairs]):
        parts.append(f"Pair {i + 1}:")
        parts.append(f"  Input ({len(p['input'])}x{len(p['input'][0])}):")
        parts.append(format_grid(p["input"]))
        parts.append(f"  Output ({len(p['output'])}x{len(p['output'][0])}):")
        parts.append(format_grid(p["output"]))
    return "\n".join(parts)


def generate_description(client: anthropic.Anthropic, task: dict, model: str) -> str:
    pairs_text = format_pairs(task)
    user_msg = USER_TEMPLATE.format(
        task_id=task["task_id"],
        n_pairs=min(3, len(task["train"])),
        pairs_text=pairs_text,
    )
    message = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        timeout=60,
    )
    return message.content[0].text.strip()


def load_existing(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save(descriptions: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(descriptions, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if description already exists")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    tasks = load_all_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    descriptions = load_existing(OUTPUT_PATH)
    already_done = set(descriptions.keys()) if not args.force else set()
    todo = [t for t in tasks if t["task_id"] not in already_done]

    print(f"Tasks total: {len(tasks)}")
    print(f"Already done: {len(already_done)}")
    print(f"To generate: {len(todo)}")
    print(f"Model: {args.model}\n")

    for i, task in enumerate(todo):
        tid = task["task_id"]
        try:
            desc = generate_description(client, task, args.model)
            descriptions[tid] = desc
            save(descriptions, OUTPUT_PATH)
            print(f"[{i + 1}/{len(todo)}] {tid}: {desc[:80]}...")
        except anthropic.RateLimitError:
            print(f"Rate limited — waiting 60s before retrying {tid}")
            time.sleep(60)
            desc = generate_description(client, task, args.model)
            descriptions[tid] = desc
            save(descriptions, OUTPUT_PATH)
        except Exception as e:
            print(f"ERROR on {tid}: {e}")
            continue

        time.sleep(args.delay)

    print(f"\nDone. {len(descriptions)} descriptions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
