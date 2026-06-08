---
name: Bottom-up sub-cluster browsing observations
description: User's observations while browsing multi-stage-rule sub-clusters task by task
type: project
originSessionId: fcc9c5ec-164b-406c-9e31-7198f453b7f9
---
## Core taxonomy principle (user's insight)
The right categorisation comes from inspecting input→output together, the same way a human
would approach a puzzle. First question: "what was done to produce the output from the input?"

### Transformation-based taxonomy (current)
- **No change / colour only** → STRUCTURE_UNCHANGED (programmatic, 52 tasks)
- **Reflection / rotation** → GEOMETRIC_TRANSFORM (programmatic, 18 tasks)
- **Translation** → TRANSLATE (programmatic, 1 task — rare in ARC)
- **Crop / extract sub-region** → CROP (programmatic, 30 tasks)
- **Input preserved + cells added** → EXTEND (programmatic, 129 tasks)
  - Straight lines added → EXTEND_LINE (33 tasks)
  - Rectangle border/fill added → EXTEND_RECT (59 tasks)
  - Everything else → 70 EXTEND-only tasks (see sub-families below)

## multi-stage-rule sub-clusters browsed

### Sub-cluster 0 (n=8)
- 253bf280: join pairs of cells on same vertical or horizontal plane
- 2dd70a9a: join pairs of cells with line avoiding cyan cells
- 56ff96f3: join pairs of cells with rectangle
- 6d58a25d: find a block; extend lines up to block and down to bottom
- 855e0971: find bands; extend line to boundary of band
- 99fa7670: find non-zero cell; extend right to border then down to bottom
- a5313dff: find zero areas surrounded by red cells; fill with blue
- af902bf9: find 4 yellow corner cells; draw red rectangle touching corners

### Sub-cluster 1 (n=16)
- 00d62c1b: find zero areas surrounded by green cells; fill with yellow
- 239be575: if cyan line joins two red squares → all cyan; else all black
- 25ff71a9: displace input down by one row (→ TRANSLATE)
- 3345333e: find rectangular block of colour; remove it
- 3428a4f5: split at yellow line; superimpose two halves; XOR
- 445eab21: find two rectangles surrounding areas; output largest-area colour block
- 54d82841: find upside-down U blocks; place yellow cell below middle
- 662c240a: unclear
- 67385a82: find 4-connected same-colour groups; colour group cyan

### User's observation on sub-clusters 0 and 1
Both contain the same two broad families — HDBSCAN didn't find a real boundary:
1. Line-extension tasks
2. Rectangle tasks

### Sub-clusters 2–6 browsed
Key additional families observed:

**Sub-cluster 2 (n=4):** move a line; repeat a pattern; fill a maze
**Sub-cluster 3 (n=4):** fill single-colour areas with pattern
**Sub-cluster 4 (n=9):** shape duplication/stamping; join diagonal cells; no change
**Sub-cluster 5 (n=4):** move blocks to fill gaps; colour swap; merge two blocks (OR)
**Sub-cluster 6 (n=26):** object movement; rotation; find unique colour; rectangle per cell;
  4-quadrant reflection; crop; diagonal extension; move to align; recolour by interior;
  fill with pattern; surround with outline; move to touch

## EXTEND-only sub-families (70 tasks not caught by EXTEND_LINE or EXTEND_RECT)

1. **Multi-direction / L-shape** (~10): 99fa7670, 6d58a25d, 178fcbfb, a2fd1cf0, d4a91cb9, dbc1a6ce, 40853293, 23581191
2. **Pattern tiling** (~8): 0dfd9992, 29ec7d0e, 484b58aa, c3f564a4, c444b776, f8c80d96
3. **Cross/stamp at markers** (~6): 4258a5f9, 913fb3ed, dc1df850, bdad9b1f, b27ca6d3
4. **Row/column propagation** (~5): d037b0a7, 1e32b0e9, d9f24cd1, 8403a5d5
5. **Rectangle-adjacent** (~6, detector gap): 56ff96f3, f35d900a, db93a21d
6. **Complex assembly** (~35): spirals, staircases, diagonal waves, nested frames
