"""
border_encoded_scale.py — Detection and solving of border-encoded scale tasks.

A task matches BORDER_ENCODED_SCALE if:
  1. Every training input is exactly 5x5.
  2. The 4x4 top-left inner region contains a coloured block on black background.
  3. The right column (rows 0-3, col 4) and bottom row (row 4, cols 0-3) plus corner
     (4,4) form an L-shaped border.
  4. Scale = number of distinct colours in the full L-border + 1.
  5. Output size = 5*scale x 5*scale.
  6. The inner 4x4 region expands to (4*scale)x(4*scale) — each input cell becomes
     scale x scale cells in the output.
  7. The border cells each expand to scale cells along the output border.
  8. Diagonal rays of colour 2 extend outward from each corner of the scaled block
     that is NOT flush with an edge of the scaled inner region, until hitting the
     inner region boundary.

Example task: 469497ad
"""

BORDER_ENCODED_SCALE_CATEGORIES = ["BORDER_ENCODED_SCALE"]


def _compute_scale(inp: list[list[int]]) -> int:
    """Compute scale from the L-border distinct colour count."""
    right_col = [inp[r][4] for r in range(4)]
    bottom_row = [inp[4][c] for c in range(4)]
    corner = inp[4][4]
    border_colours = set(right_col + bottom_row + [corner])
    return len(border_colours) + 1


def _build_output(inp: list[list[int]]) -> list[list[int]]:
    """Build the scaled output grid."""
    scale = _compute_scale(inp)
    out_h = 5 * scale
    out_w = 5 * scale
    out = [[0] * out_w for _ in range(out_h)]

    inner_size = 4 * scale  # inner region: rows 0..inner_size-1, cols 0..inner_size-1

    # 1. Scale up the inner 4x4 region
    for r in range(4):
        for c in range(4):
            colour = inp[r][c]
            for dr in range(scale):
                for dc in range(scale):
                    out[r * scale + dr][c * scale + dc] = colour

    # 2. Scale up the right border column (input col 4, rows 0-3)
    #    Each cell occupies scale rows x scale cols on the right side of the output
    for r in range(4):
        colour = inp[r][4]
        for dr in range(scale):
            for dc in range(scale):
                out[r * scale + dr][inner_size + dc] = colour

    # 3. Scale up the bottom border row (input row 4, cols 0-3)
    for c in range(4):
        colour = inp[4][c]
        for dr in range(scale):
            for dc in range(scale):
                out[inner_size + dr][c * scale + dc] = colour

    # 4. Corner cell (4,4) expands to scale x scale at bottom-right
    corner_colour = inp[4][4]
    for dr in range(scale):
        for dc in range(scale):
            out[inner_size + dr][inner_size + dc] = corner_colour

    # 5. Find the scaled block (non-zero inner cells) and draw diagonal rays
    # The scaled block bounding box within the inner region
    block_cells = [
        (r, c) for r in range(4) for c in range(4) if inp[r][c] != 0
    ]
    if block_cells:
        min_ir = min(r for r, c in block_cells)
        max_ir = max(r for r, c in block_cells)
        min_ic = min(c for r, c in block_cells)
        max_ic = max(c for r, c in block_cells)

        # Scaled block corners in output coordinates
        block_r0 = min_ir * scale          # top row of scaled block
        block_r1 = (max_ir + 1) * scale - 1  # bottom row of scaled block
        block_c0 = min_ic * scale          # left col of scaled block
        block_c1 = (max_ic + 1) * scale - 1  # right col of scaled block

        # Inner region bounds: rows 0..inner_size-1, cols 0..inner_size-1
        ir_max = inner_size - 1

        # Four corners of the scaled block and their outward diagonal directions
        corners = [
            (block_r0, block_c0, -1, -1),  # top-left → up-left
            (block_r0, block_c1, -1, +1),  # top-right → up-right
            (block_r1, block_c0, +1, -1),  # bottom-left → down-left
            (block_r1, block_c1, +1, +1),  # bottom-right → down-right
        ]

        for cr, cc, dr, dc in corners:
            # Check if corner is flush with the inner region edge
            flush_r = (cr == 0 and dr == -1) or (cr == ir_max and dr == +1)
            flush_c = (cc == 0 and dc == -1) or (cc == ir_max and dc == +1)
            if flush_r or flush_c:
                continue  # flush corner — no ray

            # Draw diagonal ray outward until hitting inner region boundary
            r, c = cr + dr, cc + dc
            while 0 <= r <= ir_max and 0 <= c <= ir_max:
                out[r][c] = 2
                r += dr
                c += dc

    return out


def detect_border_encoded_scale(task: dict) -> bool:
    """Return True if every training pair matches the border-encoded scale rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]

        if len(inp) != 5 or len(inp[0]) != 5:
            return False

        expected = _build_output(inp)
        if len(out) != len(expected) or len(out[0]) != len(expected[0]):
            return False
        if expected != out:
            return False

    return True


def solve_border_encoded_scale(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the border-encoded scale rule."""
    if len(input_grid) != 5 or len(input_grid[0]) != 5:
        return None
    return _build_output(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_border_encoded_scale(task: dict) -> list[str]:
    """Return ['BORDER_ENCODED_SCALE'] if the task matches the rule."""
    return BORDER_ENCODED_SCALE_CATEGORIES if detect_border_encoded_scale(task) else []
