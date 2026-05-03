"""
tiling.py — Detection of two related periodic tiling patterns.

TILE_FILL
  A task passes if, for every training pair:
    - input and output have the same shape
    - every non-zero cell in the input is unchanged in the output
    - every zero cell in the input becomes non-zero in the output
    - the output can be reproduced by tiling a smaller rectangular tile

  The period (tile size) can vary across training pairs; the invariant is
  that the non-zero cells of the input are always sufficient to recover the
  tile uniquely and fill all the zeros.

  Typical appearance: a large colourful grid with rectangular "holes"
  (zero patches) scattered through it; the output restores the holes.

TILE_COMPRESS
  A task passes if, for every training pair:
    - output shape divides input shape (iH % oH == 0, iW % oW == 0)
    - np.tile(output, (iH//oH, iW//oW)) == input exactly

  The scale (rh, rw) can vary across training pairs.

  Typical appearance: input is a grid repeated 2-4 times; output is the
  minimal repeating unit.
"""

import numpy as np

TILE_FILL_CATEGORIES     = ["TILE_FILL"]
TILE_COMPRESS_CATEGORIES = ["TILE_COMPRESS"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_min_period_grid(grid: np.ndarray) -> tuple[int, int]:
    """
    Return (ph, pw) — the smallest period such that:
        np.tile(grid[:ph, :pw], (rh, rw))[:H, :W] == grid
    where rh, rw are ceiling divisions.

    Returns (H, W) (the grid itself) if no smaller period exists.
    """
    H, W = grid.shape
    for ph in range(1, H + 1):
        for pw in range(1, W + 1):
            if ph == H and pw == W:
                break
            tile = grid[:ph, :pw]
            rh = (H + ph - 1) // ph
            rw = (W + pw - 1) // pw
            tiled = np.tile(tile, (rh, rw))[:H, :W]
            if np.array_equal(tiled, grid):
                return ph, pw
    return H, W


def find_period_from_nonzero(grid: np.ndarray) -> tuple[int, int] | None:
    """
    Find the minimal period (ph, pw) consistent with all non-zero cells of
    *grid* AND fully determined by them (every tile position has at least one
    non-zero cell supplying its value).

    Returns None if no such period exists (grid is all-zero, or inconsistent).
    """
    H, W = grid.shape
    nz_r, nz_c = np.where(grid != 0)
    if len(nz_r) == 0:
        return None
    vals = grid[nz_r, nz_c]

    for ph in range(1, H + 1):
        for pw in range(1, W + 1):
            tr = nz_r % ph
            tc = nz_c % pw

            # Build tile and check consistency in one pass
            tile: dict[tuple[int, int], int] = {}
            consistent = True
            for i in range(len(vals)):
                pos = (int(tr[i]), int(tc[i]))
                v   = int(vals[i])
                if pos in tile:
                    if tile[pos] != v:
                        consistent = False
                        break
                else:
                    tile[pos] = v

            if not consistent:
                continue

            # All tile positions must be covered
            if len(tile) == ph * pw:
                return ph, pw

    return None


def reconstruct_tile(grid: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """
    Build the (ph, pw) tile from the non-zero cells of *grid*.
    Assumes consistency has already been verified.
    """
    nz_r, nz_c = np.where(grid != 0)
    tile = np.zeros((ph, pw), dtype=np.int32)
    for r, c in zip(nz_r.tolist(), nz_c.tolist()):
        tile[r % ph, c % pw] = int(grid[r, c])
    return tile


# ---------------------------------------------------------------------------
# TILE_FILL detection
# ---------------------------------------------------------------------------

def detect_tile_fill(task: dict) -> bool:
    """
    Return True if every training pair matches the tile-fill pattern.

    Conditions per pair:
      1. Same shape in/out.
      2. Non-zero cells unchanged: inp[nz] == out[nz].
      3. All zero cells become non-zero.
      4. At least one zero cell exists.
      5. Output has a smaller period (it tiles from a tile < grid size).
      6. Non-zero cells of input are sufficient to uniquely determine that tile.
    """
    pairs = task.get("train", [])
    if not pairs:
        return False

    for p in pairs:
        inp = np.array(p["input"],  dtype=np.int32)
        out = np.array(p["output"], dtype=np.int32)

        if inp.shape != out.shape:
            return False

        nz = inp != 0
        z  = inp == 0

        # Condition 4: at least one zero
        if not z.any():
            return False

        # Condition 2: non-zero cells preserved
        if not np.array_equal(inp[nz], out[nz]):
            return False

        # Condition 3: all zeros become non-zero
        if np.any(out[z] == 0):
            return False

        # Condition 5: output has a period smaller than the grid
        H, W = out.shape
        ph, pw = _find_min_period_grid(out)
        if ph == H and pw == W:
            return False

        # Condition 6: non-zero cells of input uniquely determine the tile
        period = find_period_from_nonzero(inp)
        if period is None:
            return False
        # Verify the recovered tile matches the output
        tile = reconstruct_tile(inp, period[0], period[1])
        rh = (H + period[0] - 1) // period[0]
        rw = (W + period[1] - 1) // period[1]
        tiled = np.tile(tile, (rh, rw))[:H, :W]
        if not np.array_equal(tiled, out):
            return False

    return True


def categorise_tile_fill(task: dict) -> list[str]:
    """Return ['TILE_FILL'] if the task is a tile-fill pattern."""
    return TILE_FILL_CATEGORIES if detect_tile_fill(task) else []


# ---------------------------------------------------------------------------
# TILE_COMPRESS detection
# ---------------------------------------------------------------------------

def detect_tile_compress(task: dict) -> bool:
    """
    Return True if every training pair satisfies:
        np.tile(output, (iH // oH, iW // oW)) == input
    with integer scale factors > 1 in at least one dimension.
    """
    pairs = task.get("train", [])
    if not pairs:
        return False

    for p in pairs:
        inp = np.array(p["input"],  dtype=np.int32)
        out = np.array(p["output"], dtype=np.int32)
        iH, iW = inp.shape
        oH, oW = out.shape

        if oH == iH and oW == iW:
            return False  # Same size — not compression
        if iH % oH != 0 or iW % oW != 0:
            return False
        rh, rw = iH // oH, iW // oW
        if not np.array_equal(np.tile(out, (rh, rw)), inp):
            return False

    return True


def categorise_tile_compress(task: dict) -> list[str]:
    """Return ['TILE_COMPRESS'] if the task is a tile-compression pattern."""
    return TILE_COMPRESS_CATEGORIES if detect_tile_compress(task) else []
