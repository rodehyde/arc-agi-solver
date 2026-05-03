"""
Deterministic rule-based solvers for ARC task categories.

Each solver exposes a single function:
    solve(task, test_input) -> np.ndarray | None

Returns the predicted output grid, or None if the solver cannot handle this task.
"""

from .geometric import solve_geometric
from .flood_fill import solve_flood_fill
from .logical_ops import solve_logical_op

__all__ = ["solve_geometric", "solve_flood_fill", "solve_logical_op"]
