"""Tests for the geometric transform detector and solver."""

import numpy as np
import pytest

from src.categories.geometric_transforms import detect_transform
from src.solvers.geometric import solve_geometric


def _make_task(pairs, test_input):
    """Build a minimal task dict."""
    return {
        "task_id": "test",
        "train": [{"input": inp.tolist(), "output": out.tolist()} for inp, out in pairs],
        "test":  [{"input": test_input.tolist()}],
    }


# ── Detection tests ───────────────────────────────────────────────────────────

def test_detect_flip_h():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    task = _make_task([(g, g[:, ::-1])], g)
    assert detect_transform(task)[0] == "flip_h"


def test_detect_flip_v():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    task = _make_task([(g, g[::-1, :])], g)
    assert detect_transform(task)[0] == "flip_v"


def test_detect_rot180():
    g = np.array([[1, 2], [3, 4], [5, 6]])
    task = _make_task([(g, g[::-1, ::-1])], g)
    assert detect_transform(task)[0] == "rot180"


def test_detect_double_width_r():
    g = np.array([[1, 2], [3, 4]])
    out = np.concatenate([g, g[:, ::-1]], axis=1)
    task = _make_task([(g, out)], g)
    assert detect_transform(task)[0] == "double_width_r"


def test_detect_double_height_b():
    g = np.array([[1, 2, 3], [4, 5, 6]])
    out = np.concatenate([g, g[::-1, :]], axis=0)
    task = _make_task([(g, out)], g)
    assert detect_transform(task)[0] == "double_height_b"


def test_detect_quad_hv():
    g = np.array([[1, 2], [3, 4]])
    out = np.block([[g, g[:, ::-1]], [g[::-1, :], g[::-1, ::-1]]])
    task = _make_task([(g, out)], g)
    assert detect_transform(task)[0] == "quad_hv"


def test_detect_none_for_random():
    g = np.array([[1, 2], [3, 4]])
    task = _make_task([(g, np.array([[9, 9], [9, 9]]))], g)
    assert detect_transform(task) is None


def test_consistent_across_pairs():
    """All training pairs must agree — mixed transforms should return None."""
    g1 = np.array([[1, 2], [3, 4]])
    g2 = np.array([[5, 6], [7, 8]])
    task = _make_task(
        [(g1, g1[:, ::-1]), (g2, g2[::-1, :])],   # pair 1 = flip_h, pair 2 = flip_v
        g1,
    )
    assert detect_transform(task) is None


# ── Solver tests ──────────────────────────────────────────────────────────────

def test_solve_flip_h():
    g = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    task = _make_task([(g, g[:, ::-1])], g)
    pred = solve_geometric(task)
    assert np.array_equal(pred, g[:, ::-1])


def test_solve_double_width_r():
    g = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.concatenate([g, g[:, ::-1]], axis=1)
    task = _make_task([(g, expected)], g)
    pred = solve_geometric(task)
    assert np.array_equal(pred, expected)


def test_solve_returns_none_when_no_transform():
    g = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    task = _make_task([(g, np.zeros_like(g))], g)
    assert solve_geometric(task) is None


def test_solve_generalises_to_unseen_input():
    """Solver should apply the detected transform to a different test input."""
    g_train = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    g_test  = np.array([[7, 8, 9], [0, 1, 2]], dtype=np.uint8)
    task = _make_task([(g_train, g_train[:, ::-1])], g_test)
    pred = solve_geometric(task)
    assert np.array_equal(pred, g_test[:, ::-1])
