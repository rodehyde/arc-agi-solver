---
name: Current project state
description: Session 2026-06-07; ~141+/400 training tasks solved; 6 new eval-set solvers added
type: project
originSessionId: 4bd354e2-8375-4d79-869d-9a1220edc506
---
## Research discipline
- Training set (400 tasks) — train and iterate freely
- Evaluation set (400 tasks) — permanently held out benchmark; never train on it
- ARC Prize 2025 — target competition

## Current mode: batch triage (task-by-task)
Working through unsolved tasks one at a time using the 4-step protocol + decomposition pre-step.
Pace: ~1 minute per task. Stop and involve user immediately if stuck.
Write a verified solver for every task where a rule can be derived (any coverage ≥ 1).
Batch rule: 10 tasks per batch; stop at 3rd backlog entry; then review backlog with user.

## Rule-based solver coverage — ~135+/400 (2026-06-06 third session)

### Solvers added previous sessions (pre this session):
- GEOMETRIC_TRANSFORM (15), LOGICAL_OP (10), COLOUR_REMAP (6), TILE_FILL (4), COLOUR_SUBSTITUTION (4),
  COLOUR_HALO (3), QUADRANT_MIRROR (3), FLOOD_FILL_ENCLOSED (2), COLOUR_BY_HEIGHT (2), CONNECT_ALIGNED_PAIRS (2)
- Plus many coverage-1 solvers: SELF_TILE, EXTEND_PERIOD, DIAGONAL_TILE, SLIDE_TO_ADJACENT, EXPAND_CROSS,
  REPEATING_STRIPES, EXTRACT_UNIQUE_REGION, DIAGONAL_TILE_2X2, COMPLETE_REFLECTION, SEP_GRID_DIMENSIONS,
  OVERLAY_NEIGHBOURHOOD, TILE_PACK_GREY, LINE_FILL_BY_COLOUR, SNAP_TO_SEPARATOR, TWO_DOT_FRAME,
  EXTRACT_INTERIOR, ALIGN_TO_ANCHOR, CROP_BOUNDING_BOX, UNIFORM_ROW_MARK, SHIFT_DOWN_ONE,
  SEPARATOR_GRID_CROSS_FILL, ROW_FILL_MEET_MIDDLE, MIRROR_AT_MARKER, BORDER_ENCODED_SCALE,
  QUADRANT_REFLECT, RECTANGLE_FROM_CORNERS, BOUNDING_BOX_FILL, VERTICAL_COMB,
  SEPARATOR_GRID_DIAGONAL_FILL, HOLE_FILL_2X2, GAP_BRIDGE
- PARALLELOGRAM_CORRECT, STAMP_WITH_ARMS, ROTATION_COMPLETE, STAMP_ROTATED
- SEP_GRID_CONNECT, GRAVITY_DOWN, SEP_GRID_STAMP_MASTER, DIAGONAL_STAMP_EXTEND,
  PROJECT_ONTO_RECT, CONNECT_DIAGONAL_PAIRS, NEAREST_BORDER_FILL
- PATH_THROUGH_WALLS
- TILE_PERIOD_EXTRACT (2dee498d), SINGLETON_FRAME (31aa019c), STAMP_8_SHAPES (321b1fc6),
  8_BBOX_FILL (32597951), GRAVITY_TO_FLOOR (3618c87e)
- STAMP_AT_MARKER_CENTRE (363442ee), 2_BBOX_TO_4 (36fdfd69), SLIDE_UP_TO_1 (3906de3d),
  MOST_FREQUENT_SHAPE (39a8645d), L_SHAPE_COMPLETE (3aa6fb7a), MIRROR_TILE_2X2 (3af2c5a8)
- ANTIDIAG_PLUS_FLOOR (3bd67248), COMB_MIDDLE_ROW (3bdb4ada), FRAME_EXPAND_SWAP (3befdf3e)
- CORNER_FRAME_EXTRACT (3de23699), STAMP_COLOUR_ENCODED_FLIP (3e980e27),
  LARGEST_ZERO_RECT (3eda0437), FRAME_CROP_5 (3f7978a0), GRAVITY_TO_5BLOCK (4093f84a),
  FRAME_CROSSHAIR (41e4d17e), DECORATE_SHAPE_BY_2_ORIENT (36d67576)

### Solvers added this session (2026-06-06 third session):
- REMOVE_ISOLATED_SINGLETONS (42a50994): erase non-zero cells with no 8-adjacent non-zero neighbour
- HOLLOW_RECTANGLES (4347f46a): convert each solid filled rectangle to a border-only frame
- FRAME_GAP_FILL (444801d8): 1-frame with one gap side + interior marker; fill interior, gap,
  and extend a full-width line of marker colour one step beyond the gap
- SQUARE_HOLE_FILL (44d8ac46): 5-frames whose interior hole is a square (n×n) get filled with 2;
  non-square or irregular holes left unchanged
- CROSSHAIR_4FOLD_MIRROR (47c1f68c): separator cross divides grid into 4 quadrants; one quadrant
  has a shape; output removes separators and stamps shape (sep colour) into all 4 quadrants
- REPAIR_PERIODIC_TILE (484b58aa): find minimal tile period consistent with non-zero cells,
  fill all 0-patches
- MARKER_SELECT_SHAPE (48d8fb45): multiple 8-connected shapes + colour-5 marker; output is the
  shape 8-adjacent to the marker, cropped to bounding box

### Solvers added 2026-06-07 (evaluation-set triage session):
- CONCENTRIC_RINGS (516b51b7 eval): solid 1-rectangles → concentric erosion (1,2,3,2,3...)
- MATCH_RECOLOR_ONES (2a5f8217 eval): 1-shapes matched by shape to non-1 templates; replace color
- DOUBLE_DIAGONAL (f0afb749 eval): double grid; non-zero→2×2 block; \-diagonal rays→1s
- STAIRCASE_DIAGONAL (1e97544e eval): staircase diagonal pattern hole-fill
- LEGEND_ALIGN (5af49b42 eval): horizontal legend sequence aligned at isolated cell column
- FRAME_STAMP (f21745ec eval): template frame stamps pattern into same-size frames; erase others

## Policy update (2026-06-07):
User asked to inspect 10 random evaluation tasks and try to solve them.
Those 10 tasks are now no longer blind: d56f2372, 212895b5, 0a2355a6, f21745ec, 5af49b42,
516b51b7, 4c177718, 2a5f8217, f0afb749, 1e97544e.
Remaining 390 evaluation tasks still held out.

## Backlog entries (need user review):
1. `4290ef0e` — NESTED_FRAME_ASSEMBLY (concentric ring assembly, ~100+ lines)

## Next unsolved training task:
4938f0c2

## Evaluation baseline (2026-06-04)
11/400 (2.8%) on evaluation set. LOGICAL_OP: 10, SELF_TILE: 1.
Policy: run read-only periodically; never inspect individual evaluation tasks.
(Exception: 10 tasks inspected 2026-06-07 at user request.)

## Key source files
- `scripts/solvers.py` — ALL_PRIMITIVES, find_solver, verify; _connected_components helper
- `src/categories/` — individual detect/solve/categorise modules
- `results/solver_backlog.md` — tracked in git
- `CLAUDE.md` — ARC analysis protocol (4-step + 7-lens decomposition + triage pace)
