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
