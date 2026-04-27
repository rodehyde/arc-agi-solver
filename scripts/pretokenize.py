"""
pretokenize.py — Pre-encode RE-ARC examples to disk as numpy arrays.

For each RE-ARC task, encode every (input_grid, output_grid) pair into a
(T_pair, 4) int16 array (token_id, col, row, color_change — grid_number is
omitted and assigned at assembly time during training).

Output layout per task (data/tokenized/<task_id>.npz):
  train        (800, max_pair_len, 4) int16  — padded pair encodings, train split
  val          (200, max_pair_len, 4) int16  — padded pair encodings, val split
  train_lens   (800,)                 int32  — unpadded length of each train pair
  val_lens     (200,)                 int32  — unpadded length of each val pair

Pair encoding token sequence for one (input_grid, output_grid):
  START_IN  [input cells row-by-row with ROW_SEP after each row]  END_IN
  START_OUT [output cells row-by-row with ROW_SEP after each row] END_OUT

Note: no END / START / GRID_SEP tokens — those are injected when the full
in-context sequence is assembled in train_transformer.py.

Usage:
    # Encode all tasks in data/re_arc/
    python scripts/pretokenize.py

    # Encode specific task IDs only
    python scripts/pretokenize.py --tasks 007bbfb7 00d62c1b 017c7c7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.arc_tokenizer import (
    START_IN, END_IN, START_OUT, END_OUT, ROW_SEP,
    F_TOKEN, F_COL, F_ROW, F_CHANGE,
)

RE_ARC_DIR  = PROJECT_ROOT / "data" / "re_arc"
OUT_DIR     = PROJECT_ROOT / "data" / "tokenized"

N_TRAIN = 800
N_VAL   = 200


# ---------------------------------------------------------------------------
# Core encoding — one (input_grid, output_grid) pair
# ---------------------------------------------------------------------------

def encode_pair(
    inp: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Encode a single (input_grid, output_grid) pair to a (T, 4) int16 array.

    Columns: [token_id, col, row, color_change].
    grid_number (column 4 in the full 5-feature format) is NOT included here —
    it is assigned at training time when the full sequence is assembled.

    Token sequence:
      START_IN
      [input cells row-by-row; ROW_SEP after each row]
      END_IN
      START_OUT
      [output cells row-by-row; ROW_SEP after each row]
      END_OUT
    """
    tokens: list[list[int]] = []

    def _special(tok_id: int) -> list[int]:
        return [tok_id, 0, 0, 0]

    def _encode_grid(grid: np.ndarray, prev_color: int) -> tuple[list[list[int]], int]:
        H, W = grid.shape
        feats: list[list[int]] = []
        cur = prev_color
        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                change = 1 if (val != cur and cur >= 0) else 0
                cur = val
                feats.append([val, c + 1, r + 1, change])
            feats.append(_special(ROW_SEP))
        return feats, cur

    # Input grid
    tokens.append(_special(START_IN))
    grid_feats, last_color = _encode_grid(inp, prev_color=-1)
    tokens.extend(grid_feats)
    tokens.append(_special(END_IN))

    # Output grid — color_change continues from last input cell
    tokens.append(_special(START_OUT))
    grid_feats, _ = _encode_grid(out, prev_color=last_color)
    tokens.extend(grid_feats)
    tokens.append(_special(END_OUT))

    return np.array(tokens, dtype=np.int16)


# ---------------------------------------------------------------------------
# Per-task encoding
# ---------------------------------------------------------------------------

def encode_task(examples: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode all 1000 examples for one task.

    Returns:
        train        (800, max_pair_len, 4) int16
        val          (200, max_pair_len, 4) int16
        train_lens   (800,) int32
        val_lens     (200,) int32
    """
    encoded: list[np.ndarray] = []
    for ex in examples:
        inp = np.array(ex["input"],  dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        encoded.append(encode_pair(inp, out))

    train_encoded = encoded[:N_TRAIN]
    val_encoded   = encoded[N_TRAIN:]

    train_lens = np.array([e.shape[0] for e in train_encoded], dtype=np.int32)
    val_lens   = np.array([e.shape[0] for e in val_encoded],   dtype=np.int32)

    train_max = int(train_lens.max())
    val_max   = int(val_lens.max())

    # Build padded arrays — PAD rows are all zeros (token_id=0 is colour black,
    # but actual length is tracked in *_lens so padding is never consumed)
    train_arr = np.zeros((N_TRAIN, train_max, 4), dtype=np.int16)
    val_arr   = np.zeros((N_VAL,   val_max,   4), dtype=np.int16)

    for i, e in enumerate(train_encoded):
        train_arr[i, :e.shape[0]] = e
    for i, e in enumerate(val_encoded):
        val_arr[i, :e.shape[0]] = e

    return train_arr, val_arr, train_lens, val_lens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenise RE-ARC examples and save to data/tokenized/"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        metavar="TASK_ID",
        help="Task IDs to encode (8-char hex). Omit to encode all tasks in data/re_arc/.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = sorted(p.stem for p in RE_ARC_DIR.glob("*.json"))

    n_tasks = len(task_ids)
    print(f"Encoding {n_tasks} task(s) → {OUT_DIR}")

    total_bytes = 0
    for idx, task_id in enumerate(task_ids, 1):
        src = RE_ARC_DIR / f"{task_id}.json"
        if not src.exists():
            print(f"  [{idx:>3}/{n_tasks}] {task_id}  SKIP (file not found)")
            continue

        examples = json.load(open(src))
        if len(examples) < N_TRAIN + N_VAL:
            print(
                f"  [{idx:>3}/{n_tasks}] {task_id}  SKIP "
                f"(only {len(examples)} examples, need {N_TRAIN + N_VAL})"
            )
            continue

        train_arr, val_arr, train_lens, val_lens = encode_task(examples)

        out_path = OUT_DIR / f"{task_id}.npz"
        np.savez_compressed(
            out_path,
            train=train_arr,
            val=val_arr,
            train_lens=train_lens,
            val_lens=val_lens,
        )

        file_bytes = out_path.stat().st_size
        total_bytes += file_bytes
        print(
            f"  [{idx:>3}/{n_tasks}] {task_id}"
            f"  train({train_arr.shape[1]}T)  val({val_arr.shape[1]}T)"
            f"  {file_bytes / 1024:.1f} KB"
        )

    print(f"\nDone. {n_tasks} tasks, {total_bytes / (1024**2):.1f} MB total.")


if __name__ == "__main__":
    main()
