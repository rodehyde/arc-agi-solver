"""
Unit tests for src/loader.py
"""

import pytest
from src.loader import grid_dims, count_nonzero, grid_area


def test_grid_dims_square():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert grid_dims(grid) == (3, 3)


def test_grid_dims_rectangle():
    grid = [[0, 1], [2, 3], [4, 5]]
    assert grid_dims(grid) == (3, 2)


def test_grid_dims_single_row():
    grid = [[0, 0, 0, 1]]
    assert grid_dims(grid) == (1, 4)


def test_count_nonzero_all_zero():
    grid = [[0, 0], [0, 0]]
    assert count_nonzero(grid) == 0


def test_count_nonzero_mixed():
    grid = [[1, 0, 2], [0, 0, 3]]
    assert count_nonzero(grid) == 3


def test_count_nonzero_all_filled():
    grid = [[5, 5], [5, 5]]
    assert count_nonzero(grid) == 4


def test_grid_area():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert grid_area(grid) == 6


def test_grid_area_single_cell():
    grid = [[7]]
    assert grid_area(grid) == 1
