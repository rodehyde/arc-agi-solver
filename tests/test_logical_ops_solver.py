"""Tests for the logical ops detector and solver."""

import numpy as np
import pytest

from src.categories.logical_ops import detect_logical_op
from src.solvers.logical_ops import solve_logical_op


def _make_task(pairs, test_input):
    return {
        "task_id": "test",
        "train": [{"input": i.tolist(), "output": o.tolist()} for i, o in pairs],
        "test":  [{"input": test_input.tolist()}],
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _and_grid(a: np.ndarray, b: np.ndarray, fill: int) -> np.ndarray:
    mask = (a != 0) & (b != 0)
    out = np.zeros_like(a)
    out[mask] = fill
    return out


def _or_grid(a: np.ndarray, b: np.ndarray, fill: int) -> np.ndarray:
    mask = (a != 0) | (b != 0)
    out = np.zeros_like(a)
    out[mask] = fill
    return out


def _xor_grid(a: np.ndarray, b: np.ndarray, fill: int) -> np.ndarray:
    mask = (a != 0) ^ (b != 0)
    out = np.zeros_like(a)
    out[mask] = fill
    return out


def _nor_grid(a: np.ndarray, b: np.ndarray, fill: int) -> np.ndarray:
    mask = ~((a != 0) | (b != 0))
    out = np.zeros_like(a)
    out[mask] = fill
    return out


# ── detect_logical_op ─────────────────────────────────────────────────────────

def test_detect_vertical_and():
    """Two 3×3 sub-grids stacked (vertical split, no separator), AND op."""
    g1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
    g2 = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=np.int32)
    inp = np.vstack([g1, g2])         # 6×3
    out = _and_grid(g1, g2, fill=2)
    task = _make_task([(inp, out)], inp)
    result = detect_logical_op(task)
    assert result is not None
    split, op, fill = result
    assert split == "vert"
    assert op == "AND"
    assert fill == 2


def test_detect_horizontal_or():
    """Two 3×3 sub-grids side-by-side (horizontal split, no separator), OR op."""
    g1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.int32)
    g2 = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int32)
    inp = np.hstack([g1, g2])         # 3×6
    out = _or_grid(g1, g2, fill=5)
    task = _make_task([(inp, out)], inp)
    result = detect_logical_op(task)
    assert result is not None
    _, op, fill = result
    assert op == "OR"
    assert fill == 5


def test_detect_vertical_sep_xor():
    """Two 3×3 sub-grids with a separator row, XOR op."""
    g1 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.int32)
    g2 = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.int32)
    sep = np.array([[9, 9, 9]], dtype=np.int32)   # separator row
    inp = np.vstack([g1, sep, g2])    # 7×3
    out = _xor_grid(g1, g2, fill=3)
    task = _make_task([(inp, out)], inp)
    result = detect_logical_op(task)
    assert result is not None
    split, op, _ = result
    assert split == "vert_sep"
    assert op == "XOR"


def test_detect_horizontal_sep_nor():
    """Two 4×3 sub-grids with separator column, NOR op."""
    g1 = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0]], dtype=np.int32)
    g2 = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
    sep = np.full((4, 1), 9, dtype=np.int32)
    inp = np.hstack([g1, sep, g2])    # 4×7
    out = _nor_grid(g1, g2, fill=8)
    task = _make_task([(inp, out)], inp)
    result = detect_logical_op(task)
    assert result is not None
    split, op, fill = result
    assert split == "horiz_sep"
    assert op == "NOR"
    assert fill == 8


def test_detect_consistent_across_pairs():
    """Same split+op+fill detected across two training pairs."""
    g1a = np.array([[1, 0], [0, 1]], dtype=np.int32)
    g2a = np.array([[1, 1], [0, 0]], dtype=np.int32)
    inp_a = np.vstack([g1a, g2a])
    out_a = _and_grid(g1a, g2a, fill=4)

    g1b = np.array([[0, 1], [1, 0]], dtype=np.int32)
    g2b = np.array([[0, 0], [1, 1]], dtype=np.int32)
    inp_b = np.vstack([g1b, g2b])
    out_b = _and_grid(g1b, g2b, fill=4)

    task = _make_task([(inp_a, out_a), (inp_b, out_b)], inp_a)
    result = detect_logical_op(task)
    assert result is not None
    _, op, fill = result
    assert op == "AND"
    assert fill == 4


def test_detect_rejects_multiple_fill_colours():
    """Output with two non-zero colours → not a logical-op task."""
    g1 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    g2 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    inp = np.vstack([g1, g2])
    # Output has two different colours — invalid
    out = np.array([[2, 3], [0, 2]], dtype=np.int32)
    task = _make_task([(inp, out)], inp)
    assert detect_logical_op(task) is None


def test_detect_rejects_no_valid_split():
    """Input shape doesn't split evenly → None."""
    inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # 2×3
    out = np.array([[1, 2], [0, 1]], dtype=np.int32)         # 2×2 — no valid split
    task = _make_task([(inp, out)], inp)
    assert detect_logical_op(task) is None


def test_detect_rejects_no_consistent_op():
    """Two pairs disagree on which operation applies → None."""
    g1 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    g2 = np.array([[0, 1], [1, 0]], dtype=np.int32)
    inp = np.vstack([g1, g2])

    # First pair: AND
    out1 = _and_grid(g1, g2, fill=2)
    # Second pair: OR — different operation
    out2 = _or_grid(g1, g2, fill=2)

    task = _make_task([(inp, out1), (inp, out2)], inp)
    assert detect_logical_op(task) is None


# ── solve_logical_op ──────────────────────────────────────────────────────────

def test_solve_vertical_and():
    """Solver applies AND to test input split vertically."""
    g1 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    g2 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    inp = np.vstack([g1, g2])
    out = _and_grid(g1, g2, fill=3)
    task = _make_task([(inp, out)], inp)
    pred = solve_logical_op(task)
    assert pred is not None
    assert np.array_equal(pred, out)


def test_solve_horizontal_or():
    """Solver applies OR to test input split horizontally."""
    g1 = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int32)
    g2 = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int32)
    inp = np.hstack([g1, g2])
    out = _or_grid(g1, g2, fill=7)
    task = _make_task([(inp, out)], inp)
    pred = solve_logical_op(task)
    assert pred is not None
    assert np.array_equal(pred, out)


def test_solve_returns_none_when_not_detected():
    """Solver returns None for a non-logical-op task."""
    inp = np.array([[1, 2], [3, 4]], dtype=np.int32)
    task = _make_task([(inp, inp)], inp)
    assert solve_logical_op(task) is None


def test_solve_generalises_to_different_test_input():
    """Solver applies the learned op to a different test grid."""
    # Training: 2×2 grids stacked, AND, fill=5
    g1_tr = np.array([[1, 0], [0, 1]], dtype=np.int32)
    g2_tr = np.array([[1, 1], [0, 0]], dtype=np.int32)
    inp_tr = np.vstack([g1_tr, g2_tr])
    out_tr = _and_grid(g1_tr, g2_tr, fill=5)

    # Test: different 2×2 grids stacked
    g1_te = np.array([[1, 1], [0, 0]], dtype=np.int32)
    g2_te = np.array([[1, 0], [1, 0]], dtype=np.int32)
    inp_te = np.vstack([g1_te, g2_te])
    expected = _and_grid(g1_te, g2_te, fill=5)

    task = _make_task([(inp_tr, out_tr)], inp_te)
    pred = solve_logical_op(task)
    assert pred is not None
    assert np.array_equal(pred, expected)


def test_solve_with_separator_row():
    """Solver handles vert_sep split correctly on test input."""
    g1 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    g2 = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.int32)
    sep = np.full((1, 3), 9, dtype=np.int32)
    inp = np.vstack([g1, sep, g2])
    out = _xor_grid(g1, g2, fill=6)
    task = _make_task([(inp, out)], inp)
    pred = solve_logical_op(task)
    assert pred is not None
    assert np.array_equal(pred, out)


def test_solve_output_shape():
    """Solver output has the correct shape (half the input)."""
    g1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int32)
    g2 = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int32)
    inp = np.vstack([g1, g2])         # 4×4
    out = _or_grid(g1, g2, fill=2)    # 2×4
    task = _make_task([(inp, out)], inp)
    pred = solve_logical_op(task)
    assert pred is not None
    assert pred.shape == (2, 4)
