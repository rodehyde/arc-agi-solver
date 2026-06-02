"""
display.py — ARC task visualisation helpers.

Provides show_task() for use in Colab and local Jupyter notebooks.
Keeping display code here (rather than inline in the notebook) means
a git pull + re-running Cell 2 picks up changes without reopening the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from scripts.human_tree import load_task as ht_load, classify
from scripts.solvers import load_task, task_delta, find_solver, ALL_PRIMITIVES

ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
_cmap = ListedColormap(ARC_COLORS)
_norm = BoundaryNorm(boundaries=list(range(11)), ncolors=10)


def _shape_str(grid):
    return f'{grid.shape[0]}×{grid.shape[1]}'


def _show_grid(ax, grid, title='', border_colour=None):
    ax.imshow(np.array(grid, dtype=np.uint8), cmap=_cmap, norm=_norm,
              interpolation='nearest')
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    if border_colour:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border_colour)
            sp.set_linewidth(4)


def _blank_ax(ax, msg=''):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#f0f0f0')
    for sp in ax.spines.values():
        sp.set_visible(False)
    if msg:
        ax.set_title(msg, fontsize=8)


def show_task(task_id, show_prediction=True):
    """Display all training pairs then all test pairs for a task.

    Layout (one row per pair):
      col 0 : input
      col 1 : expected output  (orange border on test rows)
      col 2 : solver prediction — green ✓ / red ✗ (only when solver exists)
    """
    task  = load_task(task_id)
    train = task['train']
    tests = task['test']
    n_tr  = len(train)
    n_te  = len(tests)
    d     = task_delta(task)

    solver_name, _ = find_solver(task) if show_prediction else (None, None)
    solve_fn = None
    if solver_name:
        solve_fn = next(fn for nm, _, fn in ALL_PRIMITIVES if nm == solver_name)

    tr_preds = [None] * n_tr
    te_preds = [None] * n_te
    if solve_fn:
        try:
            r = solve_fn({**task, 'test': [{'input': p['input']} for p in train]})
            if r: tr_preds = r
        except Exception:
            pass
        try:
            r = solve_fn(task)
            if r: te_preds = r
        except Exception:
            pass

    cols   = 3 if solve_fn else 2
    n_rows = n_tr + n_te
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 3.2, n_rows * 3.2),
                             squeeze=False)

    all_tr_correct = solve_fn and all(
        p is not None and np.array_equal(p, pair['output'])
        for p, pair in zip(tr_preds, train))

    # ── Training rows ─────────────────────────────────────────────────────────
    for row, pair in enumerate(train):
        inp, out = pair['input'], pair['output']
        _show_grid(axes[row][0], inp,  f'Train {row} — input ({_shape_str(inp)})')
        _show_grid(axes[row][1], out,  f'Train {row} — expected ({_shape_str(out)})')
        if solve_fn:
            pred      = tr_preds[row]
            correct   = pred is not None and np.array_equal(pred, out)
            pred_grid = pred if pred is not None else np.zeros_like(inp)
            _show_grid(axes[row][2], pred_grid,
                       f'Train {row} — {solver_name} {"✓" if correct else "✗"} ({_shape_str(pred_grid)})',
                       border_colour='#2ECC40' if correct else '#FF4136')

    # ── Test rows ─────────────────────────────────────────────────────────────
    for i, tp in enumerate(tests):
        row     = n_tr + i
        inp     = tp['input']
        out     = tp['output']
        has_out = out is not None

        _show_grid(axes[row][0], inp, f'Test {i} — input ({_shape_str(inp)})')

        if has_out:
            _show_grid(axes[row][1], out,
                       f'Test {i} — expected ({_shape_str(out)})',
                       border_colour='#FF851B')
        else:
            _blank_ax(axes[row][1], 'no ground truth')

        if solve_fn:
            pred = te_preds[i]
            if pred is not None:
                correct_te = has_out and np.array_equal(pred, out)
                border = ('#2ECC40' if correct_te else
                          '#FF851B' if not has_out else '#FF4136')
                label  = (f'Test {i} — {solver_name} '
                          f'{"✓" if correct_te else ("(no gt)" if not has_out else "✗")} ({_shape_str(pred)})')
                _show_grid(axes[row][2], pred, label, border_colour=border)
            else:
                _blank_ax(axes[row][2], f'Test {i} — solver failed')

    # ── Title ─────────────────────────────────────────────────────────────────
    cat    = classify(ht_load(task_id))
    status = (f'  [{solver_name} — {"SOLVED" if all_tr_correct else "partial/failed"}]'
              if solve_fn else '')
    delta  = (f'gained={d["zeros_gained"]}  lost={d["zeros_lost"]}  '
              f'recoloured={d["recoloured"]}  new={d["new_colours"]}')
    fig.suptitle(f'{task_id}  [{cat}]{status}\n{delta}', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
