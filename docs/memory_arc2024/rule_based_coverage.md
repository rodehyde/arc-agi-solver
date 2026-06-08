---
name: Rule-based solver coverage
description: Verified solver count and breakdown by category (as of 2026-06-04)
type: project
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
## Coverage (as of 2026-06-04)

**68 / 400 tasks have a verified solver** (find_solver + verify passes all training pairs).

| Solver | Tasks |
|---|---|
| GEOMETRIC_TRANSFORM | 15 |
| LOGICAL_OP | 10 |
| COLOUR_REMAP | 6 |
| TILE_FILL | 4 |
| COLOUR_SUBSTITUTION | 4 |
| COLOUR_HALO | 3 |
| QUADRANT_MIRROR | 3 |
| FLOOD_FILL_ENCLOSED | 2 |
| COLOUR_BY_HEIGHT | 2 |
| CONNECT_ALIGNED_PAIRS | 2 |
| SELF_TILE | 1 |
| LOGICAL_AND | 1 |
| EXPAND_CROSS | 1 |
| LINE_FILL_BY_COLOUR | 1 |
| CROP_BOUNDING_BOX | 1 |
| UNIFORM_ROW_MARK | 1 |
| SHIFT_DOWN_ONE | 1 |
| SEPARATOR_GRID_CROSS_FILL | 1 |
| ROW_FILL_MEET_MIDDLE | 1 |
| MIRROR_AT_MARKER | 1 |
| BORDER_ENCODED_SCALE | 1 |
| QUADRANT_REFLECT | 1 |
| RECTANGLE_FROM_CORNERS | 1 |
| BOUNDING_BOX_FILL | 1 |
| VERTICAL_COMB | 1 |
| SEPARATOR_GRID_DIAGONAL_FILL | 1 |
| HOLE_FILL_2X2 | 1 |
| GAP_BRIDGE | 1 |

## Evaluation baseline (2026-06-04)
11/400 (2.8%) on evaluation set. Never inspect individual evaluation tasks.

## Source files

- `scripts/solvers.py` — ALL_PRIMITIVES registry; find_solver(); verify()
- `src/categories/` — individual detect/solve/categorise modules
- `src/categories/__init__.py` — categorise_task() aggregator

## Key implementation notes

### _make_category_solver vs _make_task_solver
Most solvers use `_make_category_solver(solve_fn)` where solve_fn takes only input_grid.
CONNECT_ALIGNED_PAIRS, GEOMETRIC_TRANSFORM, LOGICAL_OP, COLOUR_REMAP use `_make_task_solver`
because they need the full task context (training pairs) to infer colours/transforms.

### COLOUR_REMAP strategy library (TTT-style)
7 strategies tried in order: static_substitution, fill_most_common, most_common_stays,
concentric_reverse, concentric_rotate, corner_indicator, colour5_shape.
Each strategy verified against all training pairs before accepting.

### TILE_FILL solver
`find_period_from_nonzero` + `reconstruct_tile` already existed in tiling.py.
`solve_tile_fill` just applies them at test time. Period varies per pair — inferred independently.

### 73 tests passing (pytest)

## Remaining gaps (detectors exist, solvers missing or incomplete)
- GEOMETRIC_TRANSFORM: 3 tasks detected but solver fails (unhandled transform variant)
- TILE_COMPRESS: 1 detected, no solver
- CORNER_STAIRCASE: 1 detected, no solver

## Next target: SINGLE_COLOUR_OUTPUT (80 unsolved tasks)
Full list in current_state.md. TTT strategy library: "which single colour does the output show?"
Strategies to enumerate: most common, least common, unique colour, max/min region, positional.
