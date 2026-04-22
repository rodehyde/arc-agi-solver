"""
generate_descriptions.py — Use Claude to generate a natural language description
of the transformation rule for every ARC training task.

Descriptions are saved incrementally to data/descriptions_training.json so the
script can be safely interrupted and restarted.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_descriptions.py

Options:
    --limit N       Only process the first N tasks (default: all)
    --delay SECS    Seconds to wait between API calls (default: 0.5)
    --model NAME    Claude model to use (default: claude-haiku-4-5-20251001)
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
OUTPUT_PATH = PROJECT_ROOT / "data" / "descriptions_training.json"

SYSTEM_PROMPT = """\
You are an expert at analysing abstract visual puzzles. You will be shown training \
examples from an ARC (Abstraction and Reasoning Corpus) task. Each example has an \
input grid and an output grid. Colours are represented as integers 0–9.

Your job: write a concise 2–4 sentence description of the transformation rule that \
maps input to output.

Guidelines:
- Focus on the abstract rule, not specific colours (use "foreground", "background", \
"divider", "marker" etc. instead of colour numbers).
- Describe WHAT the transformation does conceptually, not how you would code it.
- If there are multiple distinct elements, describe the role of each.
- Be precise enough that someone could use your description to recognise a similar task.
- Do not start with "The transformation" — vary your opening.
"""

USER_TEMPLATE = """\
Task ID: {task_id}

Training examples ({n_pairs} pairs shown):

{pairs_text}

Describe the transformation rule in 2–4 sentences."""


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
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        timeout=30,
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
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    tasks = load_all_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    descriptions = load_existing(OUTPUT_PATH)
    already_done = set(descriptions.keys())
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
