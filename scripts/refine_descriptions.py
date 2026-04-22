"""
refine_descriptions.py — Re-generate descriptions for specific tasks using a
more precise, algorithmically-focused prompt, followed by a programmatic
verification pass that checks specific predicted cell values against the actual
grids, and a correction pass if any predictions fail.

Three-pass process:
  Pass 1 — generate structured description
  Pass 2 — model predicts specific output cell values; code checks them
  Pass 3 — if any predictions wrong, feed failures back and ask for correction

Results are saved to data/descriptions_refined.json (never overwrites the main
descriptions_training.json).

Usage:
    python scripts/refine_descriptions.py --tasks 469497ad 539a4f51 b190f7f5
    python scripts/refine_descriptions.py --tasks 469497ad --model claude-haiku-4-5-20251001
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import TextIO

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loader import load_all_tasks

PROJECT_ROOT = Path(__file__).parent.parent
ORIGINAL_PATH = PROJECT_ROOT / "data" / "descriptions_training.json"
REFINED_PATH = PROJECT_ROOT / "data" / "descriptions_refined.json"

# ---------------------------------------------------------------------------
# Pass 1 — generate structured description
# ---------------------------------------------------------------------------

PASS1_SYSTEM = """\
You are an expert at analysing abstract visual puzzles. You will be shown training \
examples from an ARC (Abstraction and Reasoning Corpus) task. Each example has an \
input grid and an output grid. Colours are represented as integers 0–9.

Your job: write a precise, algorithmic description of the transformation rule.
The description must be specific enough that someone could implement it in code.

Structure your answer in exactly this format (use these headings):

Output size: [exact relationship between output and input dimensions]

Core rule: [the precise mapping — specific enough to predict any output cell value \
given the input]

Additional rules: [any conditions, exceptions, or colour-specific behaviour — \
if none, write "None"]

Summary: [one sentence capturing the abstract concept]
"""

PASS1_USER = """\
Task ID: {task_id}

Training examples ({n_pairs} pairs):

{pairs_text}

Describe the transformation rule precisely using the required format."""

# ---------------------------------------------------------------------------
# Pass 2 — predict specific cells (model), check programmatically
# ---------------------------------------------------------------------------

PASS2_SYSTEM = """\
You are applying a proposed ARC transformation rule to predict specific output \
cell values. Output ONLY the completed prediction lines — no reasoning, no \
explanation, no other text. Replace every ? with the integer colour value (0–9) \
that your rule predicts. Start your response with the first prediction line.
"""

PASS2_USER = """\
Rule:
{description}

Training examples:
{pairs_text}

Replace every ? with the integer your rule predicts. \
Output ONLY these lines — nothing before or after:
{prediction_template}"""

# ---------------------------------------------------------------------------
# Pass 3 — correct based on specific failures
# ---------------------------------------------------------------------------

PASS3_SYSTEM = """\
You are correcting a flawed transformation rule for an ARC task. You will be \
given the original (incorrect) rule, the specific cell predictions that were \
wrong, and the full training examples.

Study the failures carefully — they tell you exactly where the rule breaks down. \
Then derive the correct rule from scratch.

Use exactly this format:

Output size: ...

Core rule: ...

Additional rules: ...

Summary: ...
"""

# ---------------------------------------------------------------------------
# Fresh start — used when correction still fails after one attempt
# ---------------------------------------------------------------------------

FRESH_SYSTEM = """\
You are analysing an ARC (Abstraction and Reasoning Corpus) task. Previous \
attempts to describe the transformation rule have all failed verification — \
they were wrong in ways that could not be fixed by patching.

Do NOT refer to any previous explanation. Forget it entirely.

Instead, look at the training examples completely fresh. Observe what changes \
between pairs. Ask yourself: does the output pattern vary between pairs or is \
it always the same shape? What role do different colours play? What determines \
the output size?

Use exactly this format:

Output size: ...

Core rule: ...

Additional rules: ...

Summary: ...
"""

FRESH_USER = """\
Previous attempts failed at these specific positions (predicted → actual):
{failures_text}

These failures suggest the conceptual model was wrong, not just the details. \
Ignore all previous explanations and derive the rule from scratch.

Training examples:
{pairs_text}

Describe the correct transformation rule."""

PASS3_USER = """\
The proposed rule was:
{description}

It failed at these specific positions (predicted → actual):
{failures_text}

Full training examples:
{pairs_text}

The failures show the rule is wrong. Study the grids carefully and write a \
corrected description."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_grid(grid: list[list[int]]) -> str:
    return "\n".join("  " + " ".join(str(c) for c in row) for row in grid)


def format_pairs(task: dict, max_pairs: int = 3) -> str:
    parts = []
    for i, p in enumerate(task["train"][:max_pairs]):
        parts.append(f"Pair {i + 1}:")
        parts.append(f"  Input ({len(p['input'])}×{len(p['input'][0])}):")
        parts.append(format_grid(p["input"]))
        parts.append(f"  Output ({len(p['output'])}×{len(p['output'][0])}):")
        parts.append(format_grid(p["output"]))
    return "\n".join(parts)


def choose_test_positions(task: dict) -> list[tuple[int, int, int]]:
    """Pick ~5 output cell positions per pair that are likely to catch rule errors.

    Returns list of (pair_idx, row, col).
    """
    positions = []
    for i, p in enumerate(task["train"][:3]):
        H = len(p["output"])
        W = len(p["output"][0])
        candidates = [
            (0, W - 1),           # top-right
            (H - 1, 0),           # bottom-left
            (H - 1, W - 1),       # bottom-right
            (H // 4, W // 4),     # interior upper-left
            (H // 4, 3 * W // 4), # interior upper-right
            (3 * H // 4, W // 4), # interior lower-left
            (H // 2, W // 2),     # centre
        ]
        seen = set()
        for r, c in candidates:
            if (r, c) not in seen:
                seen.add((r, c))
                positions.append((i, r, c))
            if len(seen) == 5:
                break
    return positions


def make_prediction_template(positions: list[tuple[int, int, int]]) -> str:
    """Return fill-in-the-blank lines like 'P1[0][8]=?' for the model to complete."""
    return "\n".join(
        f"P{pair_idx + 1}[{r}][{c}]=?"
        for pair_idx, r, c in positions
    )


def parse_predictions(text: str) -> dict[tuple[int, int, int], int]:
    """Parse lines like 'P1[2][3]=5' → {(0,2,3): 5}."""
    pattern = r"P(\d+)\[(\d+)\]\[(\d+)\]=(\d+)"
    result = {}
    for m in re.finditer(pattern, text):
        pair = int(m.group(1)) - 1
        r, c, v = int(m.group(2)), int(m.group(3)), int(m.group(4))
        result[(pair, r, c)] = v
    return result


def check_predictions(
    task: dict,
    predictions: dict[tuple[int, int, int], int],
    expected_positions: list[tuple[int, int, int]],
) -> list[str]:
    """Return list of failure strings for any wrong prediction.

    Also fails if the model produced no parseable predictions at all.
    """
    if not predictions:
        return ["Model did not produce predictions in the required format — "
                "rule could not be verified"]
    failures = []
    for (pair_idx, r, c), predicted in sorted(predictions.items()):
        if pair_idx >= len(task["train"]):
            continue
        output = task["train"][pair_idx]["output"]
        if r >= len(output) or c >= len(output[0]):
            continue
        actual = output[r][c]
        if predicted != actual:
            failures.append(
                f"Pair {pair_idx + 1} output[{r}][{c}]: "
                f"rule predicted {predicted}, actual is {actual}"
            )
    return failures


def call_api(client, model, system, user, max_tokens):
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        timeout=30,
    )
    return msg.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_description(client, task, desc, positions, template, all_pairs_text,
                        model, delay, pass_num):
    """Run the prediction check for a given description.

    Returns (failures, n_checked).
    """
    print(f"Pass {pass_num} — predicting specific cells...")
    pred_text = call_api(
        client, model, PASS2_SYSTEM,
        PASS2_USER.format(
            description=desc,
            pairs_text=all_pairs_text,
            prediction_template=template,
        ),
        max_tokens=512,
    )
    print("Predictions:")
    print(pred_text)
    time.sleep(delay)

    predictions = parse_predictions(pred_text)
    failures = check_predictions(task, predictions, positions)
    n_checked = len(predictions)
    return failures, n_checked


def correct_description(client, task, desc, failures, all_pairs_text,
                        model, delay, pass_num):
    """Generate a corrected description given specific cell failures."""
    print(f"Pass {pass_num} — correcting based on failures...")
    failures_text = "\n".join(f"  {f}" for f in failures)
    corrected = call_api(
        client, model, PASS3_SYSTEM,
        PASS3_USER.format(
            description=desc,
            failures_text=failures_text,
            pairs_text=all_pairs_text,
        ),
        max_tokens=1024,
    )
    print("CORRECTED DESCRIPTION:")
    print(corrected)
    time.sleep(delay)
    return corrected


def fresh_start_description(client, task, failures, all_pairs_text,
                            model, delay, pass_num):
    """Derive a completely fresh description, ignoring all previous attempts."""
    print(f"Pass {pass_num} — FRESH START (previous conceptual model abandoned)...")
    failures_text = "\n".join(f"  {f}" for f in failures)
    fresh = call_api(
        client, model, FRESH_SYSTEM,
        FRESH_USER.format(
            failures_text=failures_text,
            pairs_text=all_pairs_text,
        ),
        max_tokens=1024,
    )
    print("FRESH DESCRIPTION:")
    print(fresh)
    time.sleep(delay)
    return fresh


def process_task(client, task, orig_desc, model_generate, model_refine,
                 delay, sep, max_corrections=3):
    """Run the full multi-pass pipeline for one task.

    Pass 1 generates the initial description.
    Then alternates: verify (even passes) → correct (odd passes).
    Stops as soon as verification passes, or after max_corrections corrections.

    Returns (pass1_desc, passed_verification, final_desc).
    """
    tid = task["task_id"]
    pairs_text = format_pairs(task)
    all_pairs_text = format_pairs(task, max_pairs=len(task["train"]))
    positions = choose_test_positions(task)
    template = make_prediction_template(positions)

    # --- Pass 1: generate (cheap model) ---
    print(f"Pass 1 — generating description [{model_generate}]...")
    desc = call_api(
        client, model_generate, PASS1_SYSTEM,
        PASS1_USER.format(
            task_id=tid,
            n_pairs=min(3, len(task["train"])),
            pairs_text=pairs_text,
        ),
        max_tokens=1024,
    )
    print("PASS 1 DESCRIPTION:")
    print(desc)
    print(sep)
    time.sleep(delay)
    pass1_desc = desc

    # --- Alternating verify / correct loop ---
    for attempt in range(max_corrections):
        verify_pass = attempt * 2 + 2   # 2, 4, 6, ...
        correct_pass = verify_pass + 1  # 3, 5, 7, ...

        failures, n_checked = verify_description(
            client, task, desc, positions, template,
            all_pairs_text, model_refine, delay, verify_pass,
        )

        if not failures:
            print(f"  CHECK PASSED ({n_checked} cells verified).")
            return pass1_desc, True, desc

        print(f"  CHECK FAILED ({len(failures)}/{n_checked} cells wrong):")
        for f in failures:
            print(f"    ✗ {f}")

        if attempt == max_corrections - 1:
            print(f"  Max corrections reached — keeping best description so far.")
            break

        print(sep)
        if attempt == 0:
            # First correction: try patching the existing description
            desc = correct_description(
                client, task, desc, failures, all_pairs_text,
                model_refine, delay, correct_pass,
            )
        else:
            # Subsequent corrections: abandon the current frame entirely
            desc = fresh_start_description(
                client, task, failures, all_pairs_text,
                model_refine, delay, correct_pass,
            )
        print(sep)

    return pass1_desc, False, desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific task IDs to process (omit to use --all)")
    parser.add_argument("--all", action="store_true",
                        help="Process all training tasks (skips already-refined ones)")
    parser.add_argument("--redo", action="store_true",
                        help="Re-process tasks already in descriptions_refined.json")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Model for pass 1 (description generation)")
    parser.add_argument("--model-refine", default=None,
                        help="Model for correction passes (default: same as --model). "
                             "Use a more capable model here, e.g. claude-sonnet-4-6")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--log", default=None,
                        help="Write output to this file as well as stdout "
                             "(e.g. logs/refine_all.log)")
    args = parser.parse_args()

    # Set up logging to file if requested
    if args.log:
        import builtins
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "a", buffering=1)  # line-buffered

        def print(*pargs, **kwargs):  # noqa: F811
            builtins.print(*pargs, **kwargs)
            kwargs.pop("file", None)
            builtins.print(*pargs, file=log_fh, **kwargs)

    if not args.tasks and not args.all:
        print("Error: specify --tasks <id> [<id>...] or --all")
        sys.exit(1)

    client = anthropic.Anthropic()
    model_generate = args.model
    model_refine = args.model_refine or args.model
    if model_refine != model_generate:
        print(f"Pass 1 model:      {model_generate}")
        print(f"Correction model:  {model_refine}")

    all_tasks = load_all_tasks()
    task_map = {t["task_id"]: t for t in all_tasks}
    original = json.load(open(ORIGINAL_PATH)) if ORIGINAL_PATH.exists() else {}
    refined = json.load(open(REFINED_PATH)) if REFINED_PATH.exists() else {}

    if args.all:
        task_ids = [t["task_id"] for t in all_tasks]
    else:
        task_ids = args.tasks

    if not args.redo:
        already_done = set(refined.keys())
        skipping = [t for t in task_ids if t in already_done]
        task_ids = [t for t in task_ids if t not in already_done]
        if skipping:
            print(f"Skipping {len(skipping)} already-refined tasks "
                  f"(use --redo to reprocess).")

    print(f"Tasks to process: {len(task_ids)}\n")

    sep = "─" * 70
    n_corrected = 0

    for tid in task_ids:
        task = task_map.get(tid)
        if task is None:
            print(f"Task {tid} not found — skipping.")
            continue

        print(f"\n{'═' * 70}")
        print(f"Task: {tid}")
        print(sep)

        orig_val = original.get(tid, "")
        orig_desc = orig_val["description"] if isinstance(orig_val, dict) else orig_val
        print("ORIGINAL:")
        print(orig_desc)
        print(sep)

        pass1_desc, passed, final_desc = process_task(
            client, task, orig_desc,
            model_generate, model_refine, args.delay, sep
        )
        if not passed:
            n_corrected += 1

        refined[tid] = {
            "original": orig_desc,
            "pass1": pass1_desc,
            "verified": final_desc,
            "passed_verification": passed,
        }

        REFINED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REFINED_PATH, "w") as f:
            json.dump(refined, f, indent=2)

        time.sleep(args.delay)

    print(f"\n{'═' * 70}")
    print(f"Done. {len(task_ids)} tasks processed, {n_corrected} needed correction.")
    print(f"Results saved to {REFINED_PATH}")


if __name__ == "__main__":
    main()
