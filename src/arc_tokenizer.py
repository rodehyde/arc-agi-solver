"""
arc_tokenizer.py — Convert ARC grids to/from flat token sequences.

Vocabulary (18 tokens):
  0-9          : grid colours
  10  PAD      : batch padding (ignored in loss and attention)
  11  START    : beginning of sequence
  12  START_IN : beginning of an input grid
  13  END_IN   : end of an input grid
  14  START_OUT: beginning of an output grid
  15  END_OUT  : end of an output grid
  16  END      : end of sequence
  17  ROW_SEP  : end of a row within a grid

Each token position carries 5 integer features:
  [0] token_id     : colour (0-9) or special token value above
  [1] col          : 1-based column within current grid (0 for special tokens)
  [2] row          : 1-based row within current grid (0 for special tokens)
  [3] color_change : 1 if colour differs from previous colour cell, else 0
  [4] grid_number  : which grid in the sequence (0=before any grid,
                     1=ctx1 input, 2=ctx1 output, 3=ctx2 input, … etc.)

Sequence format for K context pairs + test:
  START
  START_IN [in1 row-by-row with ROW_SEP] END_IN
  START_OUT [out1 row-by-row] END_OUT
  …
  START_IN [inK] END_IN  START_OUT [outK] END_OUT
  START_IN [test_in] END_IN
  START_OUT [test_out] END

Loss is computed only on the test_out tokens (between START_OUT and END after
the test input).
"""

from __future__ import annotations

import numpy as np

# ----- vocabulary constants -----
COLORS     = 10
PAD        = 10
START      = 11
START_IN   = 12
END_IN     = 13
START_OUT  = 14
END_OUT    = 15
END        = 16
ROW_SEP    = 17
VOCAB_SIZE = 18

# Feature indices within the 5-tuple
F_TOKEN  = 0
F_COL    = 1
F_ROW    = 2
F_CHANGE = 3
F_GRID   = 4


class ArcTokenizer:
    """Encode/decode ARC grids and sequences of (input, output) pairs.

    Each token is represented as a 5-element integer array:
      [token_id, col, row, color_change, grid_number]
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _special(tok_id: int, grid_num: int) -> list[int]:
        """One special token with no spatial position."""
        return [tok_id, 0, 0, 0, grid_num]

    # ------------------------------------------------------------------
    # Single-grid encoding
    # ------------------------------------------------------------------

    def encode_grid(
        self,
        grid: np.ndarray,
        grid_number: int,
        prev_color: int = -1,
    ) -> tuple[list[list[int]], int]:
        """Flatten a 2-D uint8 grid to a list of 5-feature token rows.

        Returns:
            features   — list of [token, col, row, color_change, grid_num]
            last_color — colour of the last cell (for chaining color_change)
        """
        H, W = grid.shape
        features: list[list[int]] = []
        cur_color = prev_color

        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                change = 1 if (val != cur_color and cur_color >= 0) else 0
                cur_color = val
                features.append([val, c + 1, r + 1, change, grid_number])
            features.append(self._special(ROW_SEP, grid_number))

        return features, cur_color

    def decode_grid(self, features: list[list[int]],
                    height: int, width: int) -> np.ndarray:
        """Reconstruct a grid from a list of 5-feature rows (strips special tokens)."""
        grid = np.zeros((height, width), dtype=np.uint8)
        cells = [f[F_TOKEN] for f in features
                 if f[F_TOKEN] not in (ROW_SEP, START_IN, END_IN,
                                       START_OUT, END_OUT, PAD, START, END)]
        for idx, val in enumerate(cells[:height * width]):
            r, c = divmod(idx, width)
            grid[r, c] = min(val, 9)
        return grid

    # ------------------------------------------------------------------
    # Full sequence encoding
    # ------------------------------------------------------------------

    def encode_sequence(
        self,
        context_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input:    np.ndarray,
        test_output:   np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the full 5-feature token sequence for K context pairs + test.

        Args:
            context_pairs : list of (input_grid, output_grid) arrays
            test_input    : test input grid
            test_output   : test output grid; None at inference time

        Returns:
            features  : np.ndarray (T, 5) int16 — one row per token
            loss_mask : np.ndarray (T,) bool    — True for test_out cells only
        """
        feats: list[list[int]] = []
        loss_mask: list[bool]  = []

        def _add(tok_feats: list[list[int]], in_loss: bool) -> None:
            feats.extend(tok_feats)
            loss_mask.extend([in_loss] * len(tok_feats))

        def _special_add(tok_id: int, grid_num: int, in_loss: bool) -> None:
            _add([self._special(tok_id, grid_num)], in_loss)

        # Opening START
        _special_add(START, 0, False)

        grid_num = 0
        last_color = -1

        for inp, out in context_pairs:
            # Input grid
            grid_num += 1
            _special_add(START_IN, grid_num, False)
            g_feats, last_color = self.encode_grid(inp, grid_num, last_color)
            _add(g_feats, False)
            _special_add(END_IN, grid_num, False)

            # Output grid
            grid_num += 1
            _special_add(START_OUT, grid_num, False)
            g_feats, last_color = self.encode_grid(out, grid_num, last_color)
            _add(g_feats, False)
            _special_add(END_OUT, grid_num, False)

        # Test input (not in loss)
        grid_num += 1
        _special_add(START_IN, grid_num, False)
        g_feats, last_color = self.encode_grid(test_input, grid_num, last_color)
        _add(g_feats, False)
        _special_add(END_IN, grid_num, False)

        # Test output — these tokens are predicted
        if test_output is not None:
            grid_num += 1
            _special_add(START_OUT, grid_num, False)
            g_feats, last_color = self.encode_grid(test_output, grid_num, last_color)
            _add(g_feats, True)     # ← loss here
            _special_add(END_OUT, grid_num, True)

        # Closing END
        _special_add(END, grid_num, False)

        return np.array(feats, dtype=np.int16), np.array(loss_mask, dtype=bool)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def pad_batch(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Pad a list of encoded sequences to the same length.

        Returns numpy arrays:
            features   (B, T, 5) int16  — padded feature tensors
            loss_mask  (B, T)    bool   — True for test output tokens
            pad_mask   (B, T)    bool   — True where padding
        """
        max_len = max(f.shape[0] for f, _ in sequences)
        B = len(sequences)

        # Padding row: PAD token, all zeros
        pad_row = np.array([[PAD, 0, 0, 0, 0]], dtype=np.int16)

        out_features  = np.tile(pad_row, (B, max_len, 1))   # (B, T, 5)
        out_loss_mask = np.zeros((B, max_len), dtype=bool)
        out_pad_mask  = np.ones((B, max_len),  dtype=bool)   # True = ignore

        for i, (feats, lmask) in enumerate(sequences):
            L = feats.shape[0]
            out_features[i, :L]   = feats
            out_loss_mask[i, :L]  = lmask
            out_pad_mask[i, :L]   = False

        return {
            "features":  out_features,
            "loss_mask": out_loss_mask,
            "pad_mask":  out_pad_mask,
        }
