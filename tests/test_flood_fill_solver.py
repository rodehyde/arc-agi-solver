"""Tests for the flood fill detector and solver."""

import numpy as np
import pytest

from src.categories.flood_fill import detect_flood_fill, enclosed_background, reachable_from_boundary
from src.solvers.flood_fill import solve_flood_fill


def _make_task(pairs, test_input):
    return {
        "task_id": "test",
        "train": [{"input": i.tolist(), "output": o.tolist()} for i, o in pairs],
        "test":  [{"input": test_input.tolist()}],
    }


# ── reachable_from_boundary ───────────────────────────────────────────────────

def test_reachable_open_grid():
    """All background cells reachable when there are no enclosed regions."""
    g = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.int32)
    # The 0 in the centre is enclosed — NOT reachable from boundary
    r = reachable_from_boundary(g)
    assert not r[1, 1]


def test_reachable_border_background():
    """Background cells on the border are always reachable."""
    g = np.zeros((4, 4), dtype=np.int32)
    r = reachable_from_boundary(g)
    assert r[0, 0] and r[3, 3]


def test_enclosed_background_detects_interior():
    g = np.array([
        [3, 3, 3, 3],
        [3, 0, 0, 3],
        [3, 0, 0, 3],
        [3, 3, 3, 3],
    ], dtype=np.int32)
    enc = enclosed_background(g)
    assert enc[1, 1] and enc[1, 2] and enc[2, 1] and enc[2, 2]
    # Corners are not background so not enclosed
    assert not enc[0, 0]


# ── detect_flood_fill ─────────────────────────────────────────────────────────

def test_detect_simple():
    inp = np.array([
        [3, 3, 3],
        [3, 0, 3],
        [3, 3, 3],
    ], dtype=np.int32)
    out = inp.copy(); out[1, 1] = 4
    task = _make_task([(inp, out)], inp)
    assert detect_flood_fill(task) == 4


def test_detect_consistent_across_pairs():
    inp1 = np.array([[3,3,3],[3,0,3],[3,3,3]], dtype=np.int32)
    out1 = inp1.copy(); out1[1,1] = 7
    inp2 = np.array([[3,3,3,3],[3,0,0,3],[3,3,3,3]], dtype=np.int32)
    out2 = inp2.copy(); out2[1,1] = 7; out2[1,2] = 7
    task = _make_task([(inp1, out1), (inp2, out2)], inp1)
    assert detect_flood_fill(task) == 7


def test_detect_rejects_inconsistent_fill_colour():
    inp = np.array([[3,3,3],[3,0,3],[3,3,3]], dtype=np.int32)
    out1 = inp.copy(); out1[1,1] = 4
    out2 = inp.copy(); out2[1,1] = 5
    task = _make_task([(inp, out1), (inp, out2)], inp)
    assert detect_flood_fill(task) is None


def test_detect_rejects_non_background_change():
    """If a non-background cell changes, it's not a flood fill."""
    inp = np.array([[3,3,3],[3,1,3],[3,3,3]], dtype=np.int32)
    out = inp.copy(); out[1,1] = 4
    task = _make_task([(inp, out)], inp)
    assert detect_flood_fill(task) is None


def test_detect_rejects_open_region():
    """Changed cells reachable from the boundary → not enclosed → rejected."""
    inp = np.array([
        [3, 3, 3],
        [0, 0, 3],   # left column is open background (touches border)
        [3, 3, 3],
    ], dtype=np.int32)
    out = inp.copy(); out[1, 0] = 4; out[1, 1] = 4
    task = _make_task([(inp, out)], inp)
    assert detect_flood_fill(task) is None


def test_detect_allows_unchanged_pair():
    """A pair where output == input is allowed (no enclosed region that example)."""
    inp_fill = np.array([[3,3,3],[3,0,3],[3,3,3]], dtype=np.int32)
    out_fill = inp_fill.copy(); out_fill[1,1] = 2
    inp_none = np.array([[3,3,3],[3,3,3],[3,3,3]], dtype=np.int32)  # no enclosed region
    task = _make_task([(inp_fill, out_fill), (inp_none, inp_none.copy())], inp_fill)
    assert detect_flood_fill(task) == 2


def test_detect_rejects_shape_mismatch():
    inp = np.array([[1,1],[1,0]], dtype=np.int32)
    out = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    task = _make_task([(inp, out)], inp)
    assert detect_flood_fill(task) is None


# ── solve_flood_fill ──────────────────────────────────────────────────────────

def test_solve_fills_interior():
    inp = np.array([
        [3, 3, 3, 3],
        [3, 0, 0, 3],
        [3, 0, 0, 3],
        [3, 3, 3, 3],
    ], dtype=np.uint8)
    out = inp.copy(); out[1:3, 1:3] = 4
    task = _make_task([(inp.astype(np.int32), out.astype(np.int32))], inp.astype(np.int32))
    pred = solve_flood_fill(task)
    assert np.array_equal(pred, out)


def test_solve_returns_none_when_not_detected():
    inp = np.array([[1,2],[3,4]], dtype=np.int32)
    task = _make_task([(inp, inp)], inp)
    assert solve_flood_fill(task) is None


def test_solve_no_enclosed_region_in_test():
    """If test input has no enclosed background, output equals input."""
    train_in  = np.array([[3,3,3],[3,0,3],[3,3,3]], dtype=np.int32)
    train_out = train_in.copy(); train_out[1,1] = 5
    test_in   = np.array([[3,3,3],[3,3,3],[3,3,3]], dtype=np.int32)  # no hole
    task = _make_task([(train_in, train_out)], test_in)
    pred = solve_flood_fill(task)
    assert np.array_equal(pred, test_in)


def test_solve_generalises_to_different_test_input():
    """Solver applies fill colour from training to a different test grid."""
    train_in  = np.array([[2,2,2],[2,0,2],[2,2,2]], dtype=np.int32)
    train_out = train_in.copy(); train_out[1,1] = 6
    test_in   = np.array([[2,2,2,2],[2,0,0,2],[2,2,2,2]], dtype=np.int32)
    expected  = test_in.copy(); expected[1,1] = 6; expected[1,2] = 6
    task = _make_task([(train_in, train_out)], test_in)
    pred = solve_flood_fill(task)
    assert np.array_equal(pred, expected)
