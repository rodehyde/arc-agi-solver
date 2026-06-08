# Solver Backlog

Tasks where the rule has been identified but a solver has not yet been implemented.
Each entry records the 4-step analysis and proposed category name.

---

## `56ff96f3` — RECTANGLE_FROM_CORNERS

**Step 1:** Grid almost entirely empty; only 2–4 isolated coloured dots.  
**Step 2:** Nothing anomalous — dots are intentional seeds.  
**Step 3:** Each colour appears exactly twice; the output fills the bounding box of each same-colour pair as a solid rectangle.  
**Step 4:** Each pair of same-coloured dots marks opposite corners of a filled rectangle; fill the bounding box of each pair with that colour.

---

## `ef135b50` — GAP_BRIDGE

**Step 1:** Grid contains multiple solid rectangular blocks of 2s with gaps between them.  
**Step 2:** Gaps between adjacent blocks are conspicuously empty — they look like they should be connected.  
**Step 3:** Where two blocks share an overlapping row range and face each other horizontally (no block between them), the gap is filled with colour 9. The fill is rectangular and spans exactly the shared rows and intervening columns.  
**Step 4:** For every horizontally-adjacent pair of 2-blocks whose row spans overlap, fill the rectangular gap between their facing edges with colour 9.

---

## `272f95fa` — SEPARATOR_GRID_CROSS_FILL

**Step 1:** Colour-8 lines form a regular separator grid dividing the grid into cells.  
**Step 2:** Nothing anomalous in the separator structure.  
**Step 3:** The 8-lines always create a 3×3 array of cells. The output fills the 5 cross-shaped cells (top, left, centre, right, bottom) with fixed colours: top=2, left=4, centre=6, right=3, bottom=1. The 4 corner cells stay empty.  
**Step 4:** The 8-separator lines define a 3×3 cell grid; fill the cross-shaped cells (not corners) with positionally fixed colours — top→2, left→4, centre→6, right→3, bottom→1.

**Note:** Verify across more tasks whether fill colours are always fixed or vary per task.

---

## `6d75e8bb` — BOUNDING_BOX_FILL

**Step 1:** An irregular 8-coloured shape sits in the grid with empty space around it.  
**Step 2:** Background cells inside the bounding box of the shape stand out — outside the box is legitimately empty.  
**Step 3:** All background cells strictly inside the axis-aligned bounding box of the 8-shape are replaced with colour 2. Everything outside and the 8-cells themselves are untouched.  
**Step 4:** Fill every background cell within the axis-aligned bounding box of the 8-shape with colour 2.

---

## `a8d7556c` — HOLE_FILL_2x2

**Step 1:** Dense 18×18 grid mostly filled with 5s and scattered 0s.  
**Step 2:** Some 0-cells appear in rectangular clusters (2×2 or 3×2); isolated or L-shaped 0-clusters are left alone.  
**Step 3:** The cells that gain colour 2 are exactly those belonging to at least one 2×2 all-zero block. A 3×2 all-zero rectangle contains two overlapping 2×2 blocks, so the full 3×2 is filled — this is handled naturally by the same rule.  
**Step 4:** Find every 2×2 region that is entirely background (0); fill all cells belonging to at least one such region with colour 2.

---

## `0ca9ddb6` — COLOUR_MARKER_CROSS

**Step 1:** Sparse grid with 2–5 coloured dots, mostly empty.  
**Step 2:** Multiple colours coexist; non-1/non-2 colours (e.g. 6, 8) are never decorated in the output.  
**Step 3:** Colour 1 gains 4 orthogonal neighbours coloured 7 (+ cross). Colour 2 gains 4 diagonal neighbours coloured 4 (× cross). All other colours are untouched.  
**Step 4:** For each colour-1 cell, paint its 4 orthogonal neighbours 7; for each colour-2 cell, paint its 4 diagonal neighbours 4; leave all other cells unchanged.

---

## `8403a5d5` — VERTICAL_COMB

**Step 1:** Input is completely empty except for a single coloured cell on the bottom row.  
**Step 2:** Nothing anomalous — the single cell is the seed.  
**Step 3:** The seed column and every other column rightward are filled the full height with the seed colour. Between each consecutive pair of stripes (and one step past the last), a single colour-5 connector cell is placed, alternating between the top row and the bottom row starting at the top.  
**Step 4:** From the seed column, draw full-height vertical stripes of the seed colour at every other column rightward. Between each consecutive stripe pair (and one beyond the last), place a colour-5 cell alternating top/bottom row.

---

## `941d9a10` — SEPARATOR_GRID_DIAGONAL_FILL

**Step 1:** Regular grid of colour-5 separator lines dividing the grid into cells (all cells empty).  
**Step 2:** Nothing anomalous — the separator structure is clean and intentional.  
**Step 3:** Grid dimensions vary between pairs (3×3, 5×3, 3×5) but three cells are always filled: top-left→1, centre→2, bottom-right→3.  
**Step 4:** The colour-5 lines define an irregular grid of cells; fill the top-left cell with colour 1, the centre cell with colour 2, and the bottom-right cell with colour 3.

**Note:** Sibling of `272f95fa` (SEPARATOR_GRID_CROSS_FILL) — both use a separator grid. May belong to a broader SEPARATOR_GRID family.

---

## SEPARATOR_GRID family — allocated groups (solvers pending)

31 tasks detected with a separator grid (colour forming full-width rows + full-height columns).
Grouped into three sub-families:

### SEPARATOR_GRID_SAME_SIZE_FILL
Output is same size as input; cells within the grid get filled with colours.  
Tasks: `272f95fa`, `941d9a10`, `29623171`, `c444b776`, `bda2d7a6`, `1e32b0e9`, `39e1d7f9`, `caa06a1f`, `06df4c85`, `ed36ccf7`, `09629e4f`, `6d0160f0`, `77fdfe62`, `85c4e7cd`, `7c008303`, `9565186b`, `a68b268e`, `c59eb873`, `e48d4e1a`, `ea786f4a`, `f76d97a5`  
Rules vary by task — individual 4-step analysis needed per task.

### SEPARATOR_GRID_EXTRACT
Output is much smaller than input; the output appears to be the contents of a single selected cell extracted from the grid.  
Tasks: `2dc579da`, `6773b310`, `9f236235`, `1190e5a7`, `2dee498d`, `9af7a82c`  
Selection rule (which cell to extract) needs per-task analysis.

### SEPARATOR_GRID_EXPAND (BORDER_ENCODED_SCALE)
Output is larger than input; see `469497ad` full analysis below.  
Tasks: `469497ad`, `007bbfb7`

**Note:** `47c1f68c` (11×11→10×10) and `eb5a1d5d` (23×27→5×5) removed from this family — `47c1f68c` is a reflection task; `eb5a1d5d` has no discernible separator grid.

---

## `469497ad` — BORDER_ENCODED_SCALE

**Step 1:** Input is always 5×5: a 4×4 inner region with a coloured block on black, plus a right column and bottom row forming an L-shaped colour border — like a thumbnail encoding of something larger.  
**Step 2:** The L-border colours are structurally separate from the inner block and carry their own information.  
**Step 3:** Output sizes are 10×10, 15×15, 20×20 (scales 2×, 3×, 4×). Scale = number of distinct colours in the L-border + 1. Pairs 1 and 2 act as a legend establishing this mapping. In the output: the inner block and border both scale by the same factor. Diagonal rays of colour 2 extend outward from each free corner of the scaled inner block to the inner region boundary.  
**Step 4:** Scale = distinct colours in L-border + 1. Scale up the full 5×5 input by that factor. Place diagonal colour-2 rays outward from each corner of the scaled inner block that is not flush with the inner region edge.

**CLAUDE.md candidate:** *Input encodes its own output size via a border legend* — the L-shaped border acts as a scale key; count its distinct colours to derive the expansion factor.


---

## `47c1f68c` — QUADRANT_REFLECT

**Step 1:** Input has a separator cross (one full row + one full column of separator colour) dividing the grid into 4 quadrants. One quadrant contains a shape; the other three are empty.  
**Step 2:** The three empty quadrants are conspicuously blank — they are waiting to be filled.  
**Step 3:** The shape from the populated quadrant appears in all four quadrants of the output, recoloured to the separator colour, with the separator lines removed.  
**Step 4:** Extract the shape from the non-empty quadrant; recolour it to the separator colour; reflect it into all four quadrants (top-left: original, top-right: flip_h, bottom-left: flip_v, bottom-right: rot180); output size = 2 × quadrant dimensions.

**Note:** Distinct from existing GEOMETRIC_TRANSFORM (quad_hv) because the input is partitioned by a separator cross — the transform applies to one quadrant, not the whole grid. Needs a new solver module or extension to geometric_transforms.py.

---

## `05f2a901` — SLIDE_TO_ADJACENT (coverage = 1, awaiting family)

**Step 1:** Two shapes on a sparse background — an irregular shape (colour 2) and a compact 2×2 block (colour 8). Both overlap in either their row or column range.  
**Step 2:** The gap between them is the anomaly — shape 2 is not touching shape 8.  
**Step 3:** The 8-block is fixed. Shape 2 slides toward the 8-block along the axis where their row/column ranges overlap, until bounding boxes are adjacent (gap = 0). The shape is translated, not rotated or reshaped.  
**Step 4:** Find the axis of overlap (column ranges overlap → move vertically; row ranges overlap → move horizontally). Move shape 2 until its bounding box is one step away from shape 8's bounding box.

**Coverage note:** Rule as stated (colours 2 and 8 specifically) hits only 1 task. Likely generalises to any two distinct shapes where one slides toward the other — broaden the detect function before writing a module.

---

## `0a938d79` — REPEATING_STRIPES (coverage = 1, awaiting family)

**Step 1:** Almost entirely empty grid with exactly 2 non-zero cells; the rest is background.  
**Step 2:** Two isolated markers are conspicuously sparse — they define positions, not shapes.  
**Step 3:** Wide grid (W ≥ H) → vertical column stripes; tall grid (H > W) → horizontal row stripes. Stripes alternate between the two marker colours, starting at each marker's coordinate along the stripe axis, repeating with period = gap between markers, extending to the end of the grid. Cells before the first marker remain zero.  
**Step 4:** Find 2 marker cells. If wide → fill columns at c1, c1+gap, c1+2·gap,... alternating v1/v2 colours; if tall → same for rows.

**Coverage note:** Strict 2-marker rule hits only 1 task. Likely generalises to N markers of alternating colours with a consistent period — broaden before writing a module.

---

## `0e206a2e` — TEMPLATE_STAMP_BY_COLOUR (MOVE_TO_STATIC variant, needs family search)

**Step 1:** Sparse grid with multiple connected template shapes (cross-like structures of 8s with embedded coloured cells) alongside isolated scattered coloured cells.  
**Step 2:** Isolated non-8 coloured cells away from templates are the anomaly — detached anchor markers.  
**Step 3:** Each isolated cluster of non-8 colours (3, 1, 4) matches the non-8 colour signature of one template. The output removes all templates and stamps each template at its corresponding isolated anchor cluster, replacing the template 8s appropriately.  
**Step 4:** Match isolated colour clusters to templates by non-8 colour set; stamp template at anchor position; clear original templates.

**Coverage note:** Likely a MOVE_TO_STATIC sub-variant with multi-colour anchor matching. Needs family search before writing a module.

---

## `1caeab9d` — ALIGN_BLOCKS_TO_ANCHOR (coverage = 1, awaiting family)

**Step 1:** Sparse grid with several small same-height rectangular blocks of distinct colours, each at a different row position, spread across different columns.  
**Step 2:** The blocks are vertically misaligned — they should all share the same row band.  
**Step 3:** Exactly one block (the anchor) never moves. Every other block slides vertically, columns unchanged, until its rows match the anchor's row band. The anchor is whichever block is already at the correct row in the output.  
**Step 4:** Find the anchor block (the one whose row band is preserved); slide all other same-height blocks vertically to align with it; columns unchanged.

**Coverage note:** Rule covers only 1 task. Likely generalises to any count of same-height blocks with one fixed anchor — broaden detect before writing a module.

---

## `22233c11` — DIAGONAL_EXTEND (too hard — needs simpler framing)

**Step 1:** Sparse grid, colour-3 cells only; output keeps 3s and adds colour-8 cells.  
**Step 2:** The 8 cells are always outside the bounding box of the 3s.  
**Step 3:** Each group of 3s occupies one diagonal of its bounding box; 8s appear one further step outward from the two empty diagonal corners. For solid blocks, the group pair together occupies one diagonal of the combined bounding box and the same rule applies at group scale.  
**Step 4:** Extend the anti-diagonal outward one unit beyond the empty corners.

**Note:** The rule is correct but the multi-scale framing (individual vs combined bounding box) is brittle. Needs a simpler geometric restatement before coding. Parked until a cleaner formulation is found.

---

## `264363fd` — STAMP_WITH_EXTENDED_ARMS (MOVE_TO_STATIC variant, too complex)

**Step 1:** Large grid with distinct filled rectangles each containing a single anomalous cell, plus a small cross-shaped template in the background outside the rectangles.  
**Step 2:** Anomalous cells inside rectangles and the external template are the anomalies.  
**Step 3:** The external template is a local cross pattern centred on the anomaly colour. For each anomaly cell inside a rectangle, the template is stamped there — but its orthogonal arms extend all the way to the rectangle boundaries. The original external template disappears from the output.  
**Step 4:** Stamp template at each anomaly; extend orthogonal arms to rectangle bounds; clear original template.

**Note:** Multi-step rule requiring rectangle detection + anomaly finding + template reading + extended-arm stamping. Needs MOVE_TO_STATIC family generalisation before implementing.

---

## `2bee17df` — CORRIDOR_PASSAGE (coverage = 1, awaiting family)

**Step 1:** Grid has two distinct boundary colours with a corridor of 0-cells between them; some rows are fully open, others are constricted by infiltrating boundary cells.  
**Step 2:** The constrictions are the anomaly — the corridor is not uniform width.  
**Step 3:** Mark with colour 3: all 0-cells in fully-open rows (the wide band), and the column(s) that are 0 in every constricted row (the minimum passage).  
**Step 4:** Fill the full-width band rows entirely with 3; fill the passage column(s) — those free in every constricted row — with 3 throughout.

**Coverage note:** Rule covers only 1 task. Likely generalises beyond specific boundary colours 8 and 2 (already role-based), but needs a family search before writing a module.

---

## `3345333e` — COMPLETE_COVERED_SYMMETRY (coverage = 1, awaiting family)

**Step 1:** Grid has a shape (one colour) and a solid rectangular block (second colour) that together would form a symmetric whole.  
**Step 2:** The solid block sits exactly where the shape's reflection should be — it's covering the missing half.  
**Step 3:** Remove the block; fill those cells with the reflection of the visible shape across the symmetry axis (vertical or horizontal) defined by the output shape.  
**Step 4:** Detect: exactly 2 non-zero input colours, one disappears in output; output shape is symmetric; output = input shape ∪ its reflection, cover cells → 0.

**Coverage note:** Rule covers only 1 task. Likely a common pattern — the detector may be too strict (e.g. requiring exactly 2 colours). Consider relaxing to allow background ≠ 0 or multiple cover shapes.

---

## `9ecd008a` — RESTORE_POINT_SYMMETRY (coverage = 1, awaiting family)

**Step 1:** Large grid (16×16) is almost entirely filled with non-zero values; a small rectangular block of 0s sits somewhere in it.  
**Step 2:** The 0-block is the anomaly — the grid is otherwise regular.  
**Step 3:** Excluding the 0-block, the grid has exact 180° point symmetry about its centre. The output is the values that fill the 0-block to restore that symmetry — i.e., the values at the point-symmetric counterpart positions, read in the same top-left→bottom-right order.  
**Step 4:** Find the rectangular 0-block; verify 180° point symmetry elsewhere; output the values at (H−1−r, W−1−c) for each (r,c) in the 0-block.

**Coverage note:** Rule covers only 1 task. Likely generalises: any grid with near-perfect 180° symmetry and a rectangular hole should match. May belong to a broader RESTORE_REGULARITY family (see f9012d9b below).

---

## `f9012d9b` — RESTORE_PERIODIC_TILING (coverage = 1, awaiting family)

**Step 1:** Grid has a tiled/repeating pattern with a rectangular block of 0s in one corner or region.  
**Step 2:** The 0-block breaks the tiling — the rest of the grid is a coherent periodic pattern.  
**Step 3:** The non-zero cells have a consistent period in each column (and row). The output is the values that continue that period into the 0-block.  
**Step 4:** For each column intersecting the 0-block, determine the column's period from the non-zero cells; fill each 0 with the value at (row mod period).

**Coverage note:** Rule covers only 1 task. Likely generalises to any periodic tiling with a rectangular hole. Sibling of 9ecd008a — both are RESTORE_REGULARITY sub-variants: output fills a 0-hole to complete the grid's underlying regularity (point symmetry vs. periodic tiling).

---

## `63613498` — SEPARATOR_GRID_SHAPE_MATCH (coverage = 1, awaiting family)

**Step 1:** Input has a separator grid (single-colour lines dividing the grid into cells). One cell contains a coloured shape; the other cells contain single marker cells of various colours.  
**Step 2:** The other cells are conspicuously near-empty — each holds only one indicator cell. The shape-cell is the "answer" and the indicators are the "question."  
**Step 3:** The output shows a single cell's worth of content: the shape from the shape-cell, recoloured to match one of the indicator colours. The indicator cell selected is the one that was closest to, or structurally aligned with, the shape cell (e.g. same row or column in the separator grid).  
**Step 4:** Find the dense cell (contains a multi-cell shape); find the indicator cell that selects it (same row or column in the separator grid); output = shape recoloured to that indicator's colour, cropped to cell size.

**Note:** This task is in the COLOUR_PERMUTATION group (task IDs: `5582e5ca`, `63613498`, `85c4e7cd`, `9565186b`, `aabf363d`, `bda2d7a6`, `f76d97a5`) but its rule is structurally different from the per-cell colour remapping strategies in colour_remap.py. It involves separator grid parsing and shape extraction.

**Coverage note:** Rule covers only 1 task. Should be checked against the broader SEPARATOR_GRID family (31 tasks detected).

---

## `321b1fc6` — STAMP_TEMPLATE_AT_TARGETS (MOVE_TO_STATIC variant, coverage = 1)

**Step 1:** Small 2×2 multi-colour template sits in the top-left region; several 2×2 solid colour-8 blocks are scattered elsewhere.  
**Step 2:** The 8-blocks are conspicuously uniform — they are placeholders, not shapes.  
**Step 3:** Each 2×2 colour-8 block is replaced by the template, aligned to that block's top-left corner. The original template position becomes 0.  
**Step 4:** Locate the 2×2 template (non-8, top-left-most multi-colour block); for every 2×2 colour-8 block, stamp the template there; zero the original template cells.

**Coverage note:** Rule covers only 1 task as stated (fixed 2×2 template at 2×2 8-targets). Belongs to MOVE_TO_STATIC family — generalise to arbitrary template shape and arbitrary target colour.

---

## `3618c87e` — GRAVITY (coverage = 1)

**Step 1:** Input has a shaped region of one colour (colour 5) with scattered cells of another colour (colour 1) embedded in it.  
**Step 2:** The colour-1 cells are out of place — they float inside the 5-region.  
**Step 3:** Each colour-1 cell falls downward through the 5-region until it reaches the bottom-most row of the 5-region in its column. The 5 cells above move up to fill the vacated positions.  
**Step 4:** For each column of the 5-region, move all colour-1 cells to the bottom of the column's 5-cells (swapping 1s downward past 5s).

**Coverage note:** Rule covers only 1 task. May generalise to any direction (left/right/up) or any anchor colour.

---

## `3631a71a` — RESTORE_MASKED_REGULARITY (coverage = 1, see also 9ecd008a, f9012d9b)

**Step 1:** Large 30×30 grid is almost entirely filled with non-zero values, with scattered rectangular blobs of colour 9 throughout.  
**Step 2:** The colour-9 blobs are the anomaly — they mask cells whose values must be inferred from the underlying regular structure.  
**Step 3:** The non-9 cells have near-perfect **transposition symmetry**: inp[r,c] == inp[c,r] for all non-9 cells. Most 9s can be filled by reading inp[c,r] (the transpose counterpart). Positions where *both* (r,c) and (c,r) are 9 ("doubly-9") cannot be resolved by transposition alone — for these, rotation (e.g. rot90) may provide the answer.  
**Step 4:** For each 9-cell at (r,c): if inp[c,r] ≠ 9, output[r][c] = inp[c,r]. For doubly-9 cells, try rotated counterparts (rot90, rot270) to find a non-9 match.

**Coverage note:** Rule covers only 1 task. Belongs to RESTORE_REGULARITY family (see `9ecd008a`, `f9012d9b`). Key difficulty: the symmetry is transposition (main-diagonal reflection), not 180° point symmetry — and doubly-9 positions require a secondary symmetry (rotation) for full resolution.

**Why this was hard:** Jumped immediately to the D4 mathematical toolkit (8 symmetry transforms) rather than reading the pairs first and asking "what is the grid almost?" The data shows transposition symmetry directly — but this was only discovered after exhausting the abstract toolkit. Lesson: let the data tell you the structure before invoking a pre-formed set of transforms.

---

## `3bdb4ada` — COMB_MIDDLE_ROW (coverage = 1)

**Step 1:** Input contains multiple 3-row-tall solid filled rectangles of various colours.  
**Step 2:** The middle row of each rectangle is filled in, but it looks over-dense compared to top/bottom.  
**Step 3:** The middle row of every 3-row rectangle becomes a "comb": colour cells remain at even column indices (relative to rectangle start), zero at odd indices. Top and bottom rows unchanged.  
**Step 4:** For each 3-row solid rectangle, zero every other cell in the middle row starting from column offset 1 (keep col 0, erase col 1, keep col 2, …).

**Coverage note:** Rule covers only 1 task. The alternating pattern may be derivable from a more general rule (e.g. period-2 stripe over middle row).

---

## `42a50994` — ERASE_ISOLATED (coverage = 1)

**Step 1:** Sparse same-colour dots scattered across a large grid; some are adjacent (diagonal counts) and some are truly alone.  
**Step 2:** Isolated single-cell dots are the anomaly — they have no 8-connected neighbours.  
**Step 3:** Output retains exactly the cells that belong to connected components of size ≥ 2 (using 8-connectivity including diagonals); isolated single-cell components are erased.  
**Step 4:** Compute 8-connected components; erase all size-1 components; preserve all size ≥ 2 components unchanged.

**Coverage note:** Rule covers only 1 task. Likely generalises — search for any task where output exactly matches input with size-1 8-connected components removed.

---

## `4347f46a` — HOLLOW_RECTANGLE (coverage = 1)

**Step 1:** Input contains multiple solid filled rectangles of distinct colours on a black background.  
**Step 2:** The interiors of the rectangles are over-filled — they look solid when they should be hollow.  
**Step 3:** Each solid rectangle becomes a border-only frame: all interior cells (not on the bounding-box edge) become 0. Border cells retain their original colour.  
**Step 4:** For each solid-colour connected component whose bounding box is completely filled (a rectangle), zero all cells strictly interior to the bounding box.

**Coverage note:** Rule covers only 1 task. Likely to generalise — search for tasks where output = input with interior of each rectangle zeroed.

---

## `6e02f1e3` — DIAGONAL_SELECTOR (coverage = 1)

**Step 1:** 3×3 input (multiple pairs); each grid contains 2–3 colours filling all cells; output is a 3×3 grid with colour 5 on one diagonal and 0 elsewhere.  
**Step 2:** The "selected" diagonal (main vs. anti) varies between pairs.  
**Step 3:** One diagonal's cells form a consistent colour-sequence edge (separating the two dominant colour regions); colour 5 marks that diagonal; everything else becomes 0. The selected diagonal is the boundary between the dominant-colour upper-left triangle and the dominant-colour lower-right triangle.  
**Step 4:** Determine which diagonal is the colour-boundary (cells of dominant colour A lie on one side, dominant colour B on the other); mark those 3 diagonal cells with colour 5, zero all others.

**Coverage note:** Rule covers only 1 task. Complex selection rule — confidence MEDIUM until verified.

---

## `7e0986d6` — REPAIR_CONTAMINATED_RECTANGLES (coverage = 1)

**Step 1:** Grid contains several solid rectangles of colour 3 with occasional contaminating colour-8 cells inside them, plus isolated colour-8 cells outside any rectangle.  
**Step 2:** The colour-8 cells are anomalies — they either contaminate rectangles or float alone.  
**Step 3:** For each colour-8 cell inside a 3-rectangle, replace it with colour 3 (repair the rectangle). Isolated colour-8 cells outside any rectangle become 0.  
**Step 4:** Identify all colour-3 bounding boxes; replace every colour-8 cell within a bounding box with colour 3; erase all remaining colour-8 cells.

**Coverage note:** Rule covers only 1 task as stated (colours 3 and 8 specifically). The role-based generalisation: "repair contaminated solid rectangles by replacing the minority anomaly colour with the rectangle colour" — search for that pattern.

---

## `7f4411dc` — KEEP_SOLID_RECTANGLES (coverage = 1)

**Step 1:** Input contains a few solid rectangular blocks of one colour plus scattered individual cells and protrusions of the same colour.  
**Step 2:** The scattered individual cells and protrusions are the anomaly — they break the clean block structure.  
**Step 3:** Output retains exactly the cells that are part of an axis-aligned solid rectangle of area ≥ 4 (height ≥ 2, width ≥ 2); all other cells (isolated, single-row/column protrusions) become 0.  
**Step 4:** Enumerate all solid colour-homogeneous sub-rectangles of area ≥ 4; mark their cells; output = input where marked, 0 elsewhere.

**Coverage note:** Rule covers only 1 task. May generalise — search for tasks where keeping only cells in ≥ 2×2 solid rectangles produces the output.

---

## `855e0971` — BAND_LINE_EXTEND (coverage = 1)

**Step 1:** Grid is divided into 2–3 solid-colour rectangular bands (horizontal or vertical); each band contains exactly one or two isolated 0 cells embedded within it.  
**Step 2:** The embedded 0 cells are the anomaly — they are seeds that define a cut.  
**Step 3:** Each embedded 0 extends into a full line across its band: perpendicular to the band's long axis (0 in a horizontal band → extends to a full column; 0 in a vertical band → extends to a full row).  
**Step 4:** For each band, find all embedded 0-seed cells; extend each 0 into a complete line crossing the entire band in the short-axis direction.

**Coverage note:** Rule covers only 1 task. Likely to generalise to any grid with uniform colour bands and isolated 0-seeds.

---

## `91714a58` — EXTRACT_SOLID_BLOCK (coverage = 1)

**Step 1:** Large grid densely filled with scattered individual-coloured cells (noise), with exactly one embedded solid filled rectangle.  
**Step 2:** The solid rectangle is the only non-noise element — everything else is single isolated cells.  
**Step 3:** Output = the solid rectangle only; all noise cells (individual scattered cells not part of any solid block) are zeroed.  
**Step 4:** Find the unique connected component whose bounding box is completely filled (solid rectangle); retain it; zero everything else.

**Coverage note:** Rule covers only 1 task. Related to `7f4411dc` (KEEP_SOLID_RECTANGLES) — same idea but with a single dominant rectangle rather than multiple clean ones.

---

## `a61f2674` — COLUMN_RANK (coverage = 1)

**Step 1:** Input contains 4–5 vertical columns of same colour (colour 5) at different column positions, each of different height (counted from the bottom row).  
**Step 2:** The columns vary in height — the height difference is the key signal.  
**Step 3:** The tallest column is recoloured to 1; the shortest column is recoloured to 2; all other columns are erased (become 0).  
**Step 4:** Identify all same-colour vertical column segments; rank by height; recolour tallest → 1, shortest → 2; erase all middle-ranked columns.

**Coverage note:** Rule covers only 1 task. Likely a specific instance of a more general RANK/SELECT pattern.

---

## `d23f8c26` — KEEP_CENTER_COLUMN (coverage = 1)

**Step 1:** Input is a square (or rectangular) grid with values scattered throughout; the output has only one column non-zero.  
**Step 2:** All columns except the centre column are erased — that's the anomaly.  
**Step 3:** The output preserves exactly the values in the centre column (floor(W/2)) and zeroes everything else.  
**Step 4:** Compute centre column index = W//2; copy centre column to output; zero all other cells.

**Coverage note:** Rule covers only 1 task. Confidence MEDIUM — need to verify pair 3 and check if the rule is "centre column" or some other column selection criterion.

---

## `d89b689b` — QUADRANT_FILL_FROM_MARKERS (coverage = 1)

**Step 1:** Input contains a small solid 2×2 block (the frame) and 4 isolated single-coloured cells scattered in the four quadrants around it.  
**Step 2:** The 2×2 block's cells are all the same colour — they are placeholders for colours to be filled.  
**Step 3:** Each quadrant provides one marker cell; the corner of the 2×2 block facing that quadrant is filled with the marker's colour. The marker cells themselves are erased.  
**Step 4:** Find 2×2 solid block (the frame); find the 4 isolated marker cells (one in each quadrant relative to the block's centre); fill frame corners: top-left←TL marker, top-right←TR, bottom-left←BL, bottom-right←BR; erase all non-frame cells.

**Coverage note:** Rule covers only 1 task. Likely a specific instance of a broader FILL_FROM_CONTEXT family.

---

## `d90796e8` — ADJACENT_PAIR_MERGE (coverage = 1, confidence MEDIUM)

**Step 1:** Sparse grid with cells of 3 colours: one "survivor" (colour 5), and two "pair" colours (3 and 2) that appear near each other.  
**Step 2:** Some 3-2 adjacent pairs are visible; isolated 3s and isolated 2s also exist.  
**Step 3:** Wherever a colour-3 cell is 4-adjacent to a colour-2 cell, the pair is merged: the 3 becomes colour 8, the 2 is erased. Isolated 3s and 2s (not touching the other) survive unchanged.  
**Step 4:** For each (colour-3, colour-2) adjacent pair, replace the 3 with 8 and zero the 2. Leave non-paired cells untouched.

**Coverage note:** Rule covers only 1 task. The specific colours (3, 2 → 8) may be fixed or task-specific. Confidence MEDIUM — the rule is plausible but the output colour 8 has no clear derivation from inputs 3 and 2.

---

## `ea786f4a` — DIAGONAL_ERASE_FROM_CENTER (coverage = 1)

**Step 1:** Input is a solid colour-filled region with a single 0 cell at or near the centre.  
**Step 2:** The 0 cell is the seed; cells at diagonal positions from it look over-dense.  
**Step 3:** Every cell whose row-distance from the 0 equals its column-distance (i.e. |dr| = |dc|, both diagonals through the 0) becomes 0. All other cells retain their colour.  
**Step 4:** Find the 0 cell at (cr, cc); for every non-zero cell at (r, c) where |r−cr| = |c−cc|, set it to 0; preserve all other cells.

**Coverage note:** Rule covers only 1 task. Also listed in SEPARATOR_GRID_SAME_SIZE_FILL family — that may be a misclassification; the separator grid (if any) is not the key structure here.

---

## `228f6490` — OBJECT_INTO_FRAME (coverage = 1)

**Step 1:** Input contains one or more hollow frames (solid-border rectangles of colour 5 with hollow interiors) and several small floating objects (2×2 blocks, 1×2 strips, etc.) of various colours scattered around the grid.  
**Step 2:** The floating objects look displaced — they have no "home" in the input layout.  
**Step 3:** Each floating object's colour matches the fill that would go inside one of the frames; the output moves every floating object into the interior of its matching frame, centred or flush.  
**Step 4:** For each hollow 5-bordered frame, find the floating object whose shape/colour corresponds; place it inside the frame's interior. Erase the object from its original position.

**Coverage note:** Rule covers 1 task. The colour-matching between object and frame interior may be explicit (object colour = some frame marker) or structural. Confidence MEDIUM — full implementation needs to determine the matching criterion precisely.

---

## `09629e4f` — SEPARATOR_GRID_POSITIONAL_MAP (coverage = 1)

**Step 1:** 11×11 grid divided by colour-5 separator lines into a 3×3 array of 3×3 sub-cells. Each sub-cell either contains colour-8 (cyan) cells or content cells of other colours, but never both — except one sub-cell which has no colour-8 cells at all.  
**Step 2:** The one sub-cell without any colour-8 is anomalous — it is structurally different from all others.  
**Step 3:** That no-8 sub-cell is a positional map: each content colour appearing at position (r,c) within the 3×3 sub-cell means the output's sub-cell at position (r,c) in the 3×3 grid should be filled entirely with that colour. All other output sub-cells are blank (0). The separator lines are preserved.  
**Step 4:** Find the one 3×3 sub-cell containing no colour-8 pixels. For each (r,c) within that 3×3 that contains a non-zero, non-5 colour v, fill output sub-cell (r,c) solidly with colour v. All other sub-cells → 0.

**Verified manually against all 4 training pairs.**

**Coverage note:** Rule covers only 1 task as stated. Likely a variant of SEPARATOR_GRID_SAME_SIZE_FILL family — the no-8 sub-cell acts as a template/legend for the output layout.

---

## `36d67576` — DECORATE_SHAPE_BY_2_ORIENTATION (coverage = 1)

**Step 1:** Each input has one "complete" shape (top-left region) with 4-structure plus 1s, 2s, 3s already placed, and 2–3 "incomplete" shapes elsewhere that have the 4-structure and a 2 marker but no 1s or 3s.  
**Step 2:** The incomplete shapes are missing their decoration colours 1 and 3.  
**Step 3:** The complete shape is the template. For each incomplete shape, the 2-marker's position relative to the 4-structure's corner/bend encodes orientation. The full set of decorations from the template is applied to the incomplete shape under the D4 rotation/reflection that maps the template's 2-position to the incomplete shape's 2-position.  
**Step 4:** (1) Identify complete shape (the one containing colours 1, 2, and 3). (2) Extract its 4-skeleton and record decoration offsets relative to the bend/corner. (3) For each incomplete shape: find its corner and 2-direction; compute the D4 transform that maps template's 2-direction to this shape's 2-direction (also checking that arm directions agree); apply that transform to all decoration offsets.

**Example (pair 0):** All shapes are straight horizontal/vertical bars of length 5. 2 is always at one end, one step to the side. The decoration rule: 1s at odd distances from the 2 on the 2-side; 3s at even distances on the opposite side. Rotating the bar direction 90°/180°/270° gives the right decorations for each incomplete shape.

**Coverage note:** Rule verified verbally across all 3 pairs. Solver NOT implemented — needs D4 transform logic plus shape-skeleton extraction (handles bars, L-shapes, plus-shapes). Estimated ~80 lines. Medium implementation complexity.

---

## `4290ef0e` — NESTED_FRAME_ASSEMBLY (coverage = 1)

**Step 1:** Input contains multiple colored shapes, each forming a partial rectangular "corner-cluster" pattern. One color may appear as a single isolated cell (the center marker).
**Step 2:** The shapes are scattered around the input with no clear spatial relationship. Each shape is a partial frame showing corner clusters and/or edge segments.
**Step 3:** The shapes, when ordered by bounding box size, form concentric rings of a nested symmetric (D4) frame. The output is assembled by placing each shape's corner pattern into the appropriate ring, from largest (outermost) to smallest (innermost), with the isolated single cell (if any) as the center.
**Step 4:** (1) Extract each colored shape's bounding box size. (2) Sort shapes by area, largest first. (3) Extract the "corner cluster" pattern from each shape (top-left quarter, then mirror 4-fold). (4) Build output grid starting with the largest shape's bounding box; place each ring concentrically inward.

**Example (pair 0):** 3 shapes: 6-shape (7×7), 1-shape (5×5), 3-shape (3×3). Output 7×7 has 6-ring (corners), 1-ring (inner L-clusters), 3-ring (center 3×3 area).

**Coverage note:** Rule verified verbally. Solver NOT implemented — complex to build correctly (corner-cluster extraction, 4-fold symmetry reconstruction, ring placement). The CENTER cell (if present) fills the very center. High implementation complexity (~100+ lines). Confidence HIGH on the rule.

---

## `212895b5` — BLOCK_CORNER_RAYS (evaluation set, closeness=1)

**Step 1:** Large grid of 0s with a 3×3 block of 8s and many scattered 5s.  
**Step 2:** Nothing anomalous in input structure — 5s are isolated markers.  
**Step 3 (partial):** The output adds 2s and 4s. The 2s form diagonal rays from each corner of the 3×3 block (NW from TL, NE from TR, SW from BL, SE from BR). Each ray continues until hitting a 5 (stops at the cell just before the 5) or the grid boundary. The 4s form orthogonal arm patterns from the block edges, but the exact 4-rule is unclear.

**4-ray rule (verified pair 1 and pair 2 cleanly):** From each block corner, shoot a diagonal ray one step at a time. Place a 2 at each cell until the next step would land on a 5, or the grid boundary is reached.

**4-arm rule (NOT yet understood):**  
- In pair 1 the horizontal arms from the block middle row are 1 cell each side (stopping before adjacent 5s).  
- The upper-right 4-cluster (pair 1) forms an alternating wide/narrow diagonal pattern toward the NE-direction 5 group.  
- Pattern: 4s appear 1–3 cols to the "inside" of the NE 2-ray position at each row — alternating single cell and 3-cell-wide strips.  
- Pair 0 anomaly: TL 2-ray stops 3 cells from the block corner instead of 6 cells (expected 5 cells before the boundary 5). Possibly blocked by cluster of adjacent 5s at rows 1-3.  

**Step 4 (partial):** For each block corner, shoot a diagonal ray outward placing 2s until a 5 or grid edge is hit. Then apply a 4-arm rule from each block face center (orthogonal extension until a 5 or grid edge). The 4 rule involves the 3-unit width of the block projecting outward, creating alternating single and triple-wide bands — but the precise rule is unclear.

**Coverage note:** 2-ray rule coded mentally but NOT registered. 4-arm rule NOT understood. Needs more analysis.
