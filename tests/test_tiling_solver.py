"""Tests for the tiling detectors and solvers (TILE_FILL and TILE_COMPRESS)."""

import numpy as np
import pytest

from src.categories.tiling import (
    detect_tile_fill,
    detect_tile_compress,
    find_period_from_nonzero,
    reconstruct_tile,
)
from src.solvers.tiling import solve_tile_fill, solve_tile_compress


def _make_task(pairs, test_input):
    return {
        "task_id": "test",
        "train": [{"input": i.tolist(), "output": o.tolist()} for i, o in pairs],
        "test":  [{"input": test_input.tolist()}],
    }


# ── find_period_from_nonzero ──────────────────────────────────────────────────

def test_find_period_simple_2x2():
    """Sparse 4×4 grid with 2×2 period detected from non-zero cells."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    grid = np.tile(tile, (2, 2))  # fully tiled, no zeros
    ph, pw = find_period_from_nonzero(grid)
    assert (ph, pw) == (2, 2)


def test_find_period_sparse_cells():
    """Period found even when most cells are zero (many holes)."""
    # Tile [[5, 3], [3, 5]]: tile[r%2, c%2]
    # tile(0,0)=5, tile(0,1)=3, tile(1,0)=3, tile(1,1)=5
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[0, 0] = 5   # (0%2, 0%2) = (0,0) → 5
    grid[0, 2] = 5   # (0%2, 2%2) = (0,0) → 5
    grid[0, 1] = 3   # (0%2, 1%2) = (0,1) → 3
    grid[1, 0] = 3   # (1%2, 0%2) = (1,0) → 3
    grid[1, 1] = 5   # (1%2, 1%2) = (1,1) → 5
    grid[4, 2] = 5   # (4%2, 2%2) = (0,0) → 5
    result = find_period_from_nonzero(grid)
    assert result is not None
    ph, pw = result
    assert ph <= 2 and pw <= 2


def test_find_period_returns_none_for_all_zeros():
    grid = np.zeros((4, 4), dtype=np.int32)
    assert find_period_from_nonzero(grid) is None


def test_find_period_inconsistent():
    """Conflicting values at same tile position → no period found (or larger period)."""
    grid = np.zeros((4, 4), dtype=np.int32)
    # Two cells at (0,0) and (2,0) with period 2 would both map to (0,0)
    # but have different values → period-2 is rejected
    grid[0, 0] = 1
    grid[2, 0] = 2   # conflicts if period=2 (both → (0,0))
    grid[1, 0] = 3
    result = find_period_from_nonzero(grid)
    # Either returns a larger period or None; if a period is found,
    # (0,0) and (2,0) must NOT be the same tile position
    if result is not None:
        ph, pw = result
        assert (0 % ph, 0 % pw) != (2 % ph, 0 % pw) or grid[0, 0] == grid[2, 0]


# ── detect_tile_fill ──────────────────────────────────────────────────────────

def test_detect_tile_fill_basic():
    """Simple 6×6 grid with period-3 tile and holes."""
    tile = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    out = np.tile(tile, (2, 2))
    inp = out.copy()
    # Punch some holes
    inp[0, 1] = 0
    inp[3, 4] = 0
    inp[5, 0] = 0
    task = _make_task([(inp, out)], inp)
    assert detect_tile_fill(task)


def test_detect_tile_fill_multiple_pairs():
    """Two pairs with different tile sizes both detected."""
    tile_a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out_a  = np.tile(tile_a, (3, 3))    # 6×6
    inp_a  = out_a.copy(); inp_a[0, 1] = 0; inp_a[4, 2] = 0

    tile_b = np.array([[5, 6, 7], [8, 9, 1]], dtype=np.int32)
    out_b  = np.tile(tile_b, (2, 2))[:4, :6]    # 4×6
    inp_b  = out_b.copy(); inp_b[1, 0] = 0

    task = _make_task([(inp_a, out_a), (inp_b, out_b)], inp_a)
    assert detect_tile_fill(task)


def test_detect_tile_fill_rejects_no_zeros():
    """No zeros in input → not a tile-fill task."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out  = np.tile(tile, (2, 2))
    task = _make_task([(out.copy(), out)], out)
    assert not detect_tile_fill(task)


def test_detect_tile_fill_rejects_shape_mismatch():
    """Different input/output shapes → rejected."""
    inp = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    out = np.array([[1, 1], [1, 1]], dtype=np.int32)
    task = _make_task([(inp, out)], inp)
    assert not detect_tile_fill(task)


def test_detect_tile_fill_rejects_nonzero_changed():
    """A non-zero cell that changes value → rejected."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out  = np.tile(tile, (2, 2))
    inp  = out.copy(); inp[0, 0] = 0
    # Corrupt: the (0,0) cell in output is different from expected
    bad_out = out.copy(); bad_out[2, 2] = 99
    task = _make_task([(inp, bad_out)], inp)
    assert not detect_tile_fill(task)


def test_detect_tile_fill_rejects_output_with_zeros():
    """Output still has zeros → not fully filled → rejected."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out  = np.tile(tile, (2, 2))
    inp  = out.copy(); inp[0, 1] = 0
    bad_out = out.copy(); bad_out[1, 3] = 0   # output not complete
    task = _make_task([(inp, bad_out)], inp)
    assert not detect_tile_fill(task)


def test_detect_tile_fill_rejects_no_period():
    """Output has no internal period (itself is minimal) → rejected."""
    inp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
    out = np.array([[9, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
    task = _make_task([(inp, out)], inp)
    assert not detect_tile_fill(task)


# ── detect_tile_compress ──────────────────────────────────────────────────────

def test_detect_tile_compress_horizontal():
    """Input is tile repeated twice horizontally."""
    tile = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    inp  = np.tile(tile, (1, 2))   # 2×6
    task = _make_task([(inp, tile)], inp)
    assert detect_tile_compress(task)


def test_detect_tile_compress_vertical():
    """Input is tile repeated twice vertically."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    inp  = np.tile(tile, (2, 1))   # 4×2
    task = _make_task([(inp, tile)], inp)
    assert detect_tile_compress(task)


def test_detect_tile_compress_both_dims():
    """Input is tile repeated 3×3."""
    tile = np.array([[5, 6], [7, 8]], dtype=np.int32)
    inp  = np.tile(tile, (3, 3))   # 6×6
    task = _make_task([(inp, tile)], inp)
    assert detect_tile_compress(task)


def test_detect_tile_compress_varying_scale():
    """Scale can vary across training pairs."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    inp_a = np.tile(tile, (2, 1))   # 4×2
    inp_b = np.tile(tile, (1, 3))   # 2×6
    task = _make_task([(inp_a, tile), (inp_b, tile)], inp_a)
    assert detect_tile_compress(task)


def test_detect_tile_compress_rejects_same_size():
    """Same input/output size → not a compression."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    task = _make_task([(inp, inp.copy())], inp)
    assert not detect_tile_compress(task)


def test_detect_tile_compress_rejects_non_divisible():
    """Output size doesn't divide input size → rejected."""
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)   # 2×3
    out = np.array([[1, 2]], dtype=np.int32)                   # 1×2 — 3 % 2 != 0
    task = _make_task([(inp, out)], inp)
    assert not detect_tile_compress(task)


def test_detect_tile_compress_rejects_non_tiling():
    """Output doesn't reproduce input when tiled → rejected."""
    inp = np.array([[1, 2, 3, 4]], dtype=np.int32)   # 1×4
    out = np.array([[1, 2]], dtype=np.int32)          # 1×2 — tile would give [1,2,1,2] ≠ [1,2,3,4]
    task = _make_task([(inp, out)], inp)
    assert not detect_tile_compress(task)


# ── solve_tile_fill ───────────────────────────────────────────────────────────

def test_solve_tile_fill_fills_holes():
    """Solver correctly fills zero holes using the period from non-zero cells."""
    tile = np.array([[2, 3], [5, 7]], dtype=np.int32)
    out  = np.tile(tile, (3, 3))    # 6×6
    inp  = out.copy()
    inp[0, 0] = 0; inp[2, 4] = 0; inp[5, 1] = 0
    task = _make_task([(inp, out)], inp)
    pred = solve_tile_fill(task)
    assert pred is not None
    assert np.array_equal(pred, out)


def test_solve_tile_fill_no_holes_returns_input():
    """Test input with no holes → output equals input unchanged."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out  = np.tile(tile, (2, 2))
    inp  = out.copy()
    # Training pair has holes, test input doesn't
    inp_tr = inp.copy(); inp_tr[0, 0] = 0
    task = _make_task([(inp_tr, out)], inp)   # test has no zeros
    pred = solve_tile_fill(task)
    assert pred is not None
    assert np.array_equal(pred, inp)


def test_solve_tile_fill_generalises_to_different_test():
    """Solver works when test period differs from training periods."""
    # Training: 2×2 tile in a 4×4 grid
    tile_tr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out_tr  = np.tile(tile_tr, (2, 2))
    inp_tr  = out_tr.copy(); inp_tr[1, 0] = 0

    # Test: 3×3 tile in a 6×6 grid
    tile_te = np.array([[5, 6, 7], [8, 9, 1], [2, 3, 4]], dtype=np.int32)
    out_te  = np.tile(tile_te, (2, 2))
    inp_te  = out_te.copy(); inp_te[0, 1] = 0; inp_te[3, 4] = 0; inp_te[5, 2] = 0

    task = _make_task([(inp_tr, out_tr)], inp_te)
    pred = solve_tile_fill(task)
    assert pred is not None
    assert np.array_equal(pred, out_te)


def test_solve_tile_fill_returns_none_when_not_detected():
    """Returns None for a non-tile-fill task."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    task = _make_task([(inp, inp.copy())], inp)
    assert solve_tile_fill(task) is None


# ── solve_tile_compress ───────────────────────────────────────────────────────

def test_solve_tile_compress_extracts_tile():
    """Solver returns the minimal repeating tile."""
    tile = np.array([[3, 7], [8, 2]], dtype=np.int32)
    inp  = np.tile(tile, (3, 2))   # 6×4
    task = _make_task([(inp, tile)], inp)
    pred = solve_tile_compress(task)
    assert pred is not None
    assert np.array_equal(pred, tile)


def test_solve_tile_compress_generalises():
    """Solver extracts correct tile from unseen test input."""
    tile = np.array([[1, 2], [3, 4]], dtype=np.int32)
    inp_tr = np.tile(tile, (2, 1))   # 4×2
    inp_te = np.tile(tile, (3, 2))   # 6×4

    task = _make_task([(inp_tr, tile)], inp_te)
    pred = solve_tile_compress(task)
    assert pred is not None
    assert np.array_equal(pred, tile)


def test_solve_tile_compress_returns_none_when_not_detected():
    """Returns None for a non-tile-compress task."""
    inp = np.array([[1, 2, 3, 4]], dtype=np.int32)   # no repeating tile
    out = np.array([[1, 2]], dtype=np.int32)
    task = _make_task([(inp, out)], inp)
    assert solve_tile_compress(task) is None
