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

Your goal: write a description that captures the MECHANISM — the computational rule
that transforms the input into the output. Focus entirely on the PROCESS, not the
appearance of the result. Two tasks whose outputs look similar may have completely
different rules; your description must distinguish them.

Critical warning: do NOT describe what the output looks like. Describe what
operations the rule performs on the input. If the output "looks like a grid of
compartments", that is an appearance — ask instead: does the rule DISCOVER those
compartments from the input, or are they already given by fixed lines? Does it
FILL based on majority vote, flood-fill reachability, copying a shape, or
something else? Name the computational primitive.

Work through this exact sequence:

1. SCENE: What does each input look like as a scene? Name the role of each colour.
   Identify what is structural (fixed lines, borders, markers) vs what is content
   (the data the rule reads to decide the output). Do NOT describe the output yet.

2. INVARIANTS: What is the same across ALL input/output pairs? (Fixed structure,
   which colours always appear, any spatial layout that never changes?)

3. VARIABLES: What differs between examples? (Sizes, positions, colours, counts?)

4. MECHANISM: Describe the rule as a sequence of operations the rule executes:
   (a) What does the rule READ from the input? (specific cells, counts, positions,
       shapes, colours — be precise about what information is consumed)
   (b) What OPERATION does it perform? Name the computational primitive:
       e.g. majority-vote, flood-fill from boundary, copy-and-place, extend-line,
       reflect, count-and-output, sort, select-one, fill-solid.
   (c) What does it WRITE to the output and where?
   Do not describe what the output looks like — describe the rule's actions.

5. TYPE: 1–4 words naming the core computational operation — use a verb.
   Good: "majority vote fill", "boundary flood fill", "shape copy propagation",
         "line extension from seeds", "region count output".
   Bad: "compartment fill" (appearance, not operation), "colour task" (too vague).

Output exactly these five labelled lines and nothing else:
SCENE: <1-2 sentences>
INVARIANTS: <1-2 sentences>
VARIABLES: <1 sentence>
MECHANISM: <3-4 sentences covering read / operation / write>
TYPE: <1-4 words, verb-first>"""

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
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/descriptions_scene.json)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if description already exists")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    output_path = Path(args.output) if args.output else OUTPUT_PATH

    tasks = load_all_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    descriptions = load_existing(output_path)
    already_done = set(descriptions.keys()) if not args.force else set()
    todo = [t for t in tasks if t["task_id"] not in already_done]

    print(f"Tasks total: {len(tasks)}")
    print(f"Already done: {len(already_done)}")
    print(f"To generate: {len(todo)}")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}\n")

    for i, task in enumerate(todo):
        tid = task["task_id"]
        try:
            desc = generate_description(client, task, args.model)
            descriptions[tid] = desc
            save(descriptions, output_path)
            print(f"[{i + 1}/{len(todo)}] {tid}: {desc[:80]}...")
        except anthropic.RateLimitError:
            print(f"Rate limited — waiting 60s before retrying {tid}")
            time.sleep(60)
            desc = generate_description(client, task, args.model)
            descriptions[tid] = desc
            save(descriptions, output_path)
        except Exception as e:
            print(f"ERROR on {tid}: {e}")
            continue

        time.sleep(args.delay)

    print(f"\nDone. {len(descriptions)} descriptions saved to {output_path}")


if __name__ == "__main__":
    main()
