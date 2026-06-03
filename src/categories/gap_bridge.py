"""
gap_bridge.py — Detection and solving of gap-bridge tasks.

A task matches GAP_BRIDGE if:
  1. Every training input consists of solid rectangular blocks of colour 2 on a
     background of 0.
  2. For every horizontally-adjacent pair of 2-blocks whose row spans overlap
     and no other block lies between them in the column direction, the output
     fills the rectangular gap between their facing edges with colour 9.
  3. The fill spans exactly the shared row range and the columns between the blocks.

Example task: ef135b50
"""

GAP_BRIDGE_CATEGORIES = ["GAP_BRIDGE"]


def _find_blocks(inp: list[list[int]]) -> list[tuple[int, int, int, int]]:
    """
    Find all maximal connected rectangular blocks of colour 2.
    Returns list of (min_r, max_r, min_c, max_c).
    """
    H, W = len(inp), len(inp[0])
    visited = [[False] * W for _ in range(H)]
    blocks = []

    for r in range(H):
        for c in range(W):
            if inp[r][c] == 2 and not visited[r][c]:
                # BFS to find connected component
                min_r, max_r, min_c, max_c = r, r, c, c
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and inp[nr][nc] == 2:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                blocks.append((min_r, max_r, min_c, max_c))

    return blocks


def _apply_bridges(inp: list[list[int]]) -> list[list[int]]:
    """
    Apply the gap-bridge rule: for each pair of 2-blocks with overlapping row spans
    and no intervening block, fill the gap with colour 9.
    """
    H, W = len(inp), len(inp[0])
    out = [row[:] for row in inp]
    blocks = _find_blocks(inp)

    # Sort blocks by their left edge for column-order comparison
    blocks_sorted = sorted(blocks, key=lambda b: b[2])  # sort by min_c

    for i in range(len(blocks_sorted)):
        for j in range(i + 1, len(blocks_sorted)):
            b1 = blocks_sorted[i]
            b2 = blocks_sorted[j]
            # b1 is to the left of b2 (or same column)
            # Check if they are horizontally adjacent (b1 left of b2 with a gap)
            if b1[3] >= b2[2]:
                # Overlapping or touching in columns — not a gap
                continue

            # Check row span overlap
            row_lo = max(b1[0], b2[0])
            row_hi = min(b1[1], b2[1])
            if row_lo > row_hi:
                continue  # no row overlap

            # Check no other block lies between them in the column range
            gap_c_lo = b1[3] + 1
            gap_c_hi = b2[2] - 1
            if gap_c_lo > gap_c_hi:
                continue  # blocks are adjacent with no gap

            blocked = False
            for k, bk in enumerate(blocks_sorted):
                if k == i or k == j:
                    continue
                # Does bk lie between b1 and b2 in the column direction?
                if bk[2] <= b1[3] or bk[3] >= b2[2]:
                    continue
                # bk is in the column range; check if it overlaps the row range
                if bk[1] >= row_lo and bk[0] <= row_hi:
                    blocked = True
                    break

            if blocked:
                continue

            # Fill the gap
            for r in range(row_lo, row_hi + 1):
                for c in range(gap_c_lo, gap_c_hi + 1):
                    out[r][c] = 9

    return out


def detect_gap_bridge(task: dict) -> bool:
    """Return True if every training pair matches the gap-bridge rule."""
    for p in task["train"]:
        inp = p["input"]
        out = p["output"]
        H, W = len(inp), len(inp[0])

        if len(out) != H or len(out[0]) != W:
            return False

        # Input must only contain 0s and 2s
        for r in range(H):
            for c in range(W):
                if inp[r][c] not in (0, 2):
                    return False

        expected = _apply_bridges(inp)
        if expected != out:
            return False

    return True


def solve_gap_bridge(input_grid: list[list[int]]) -> list[list[int]] | None:
    """Apply the gap-bridge rule to fill gaps between 2-blocks with colour 9."""
    H, W = len(input_grid), len(input_grid[0])
    for r in range(H):
        for c in range(W):
            if input_grid[r][c] not in (0, 2):
                return None
    return _apply_bridges(input_grid)


# ---------------------------------------------------------------------------
# Category interface
# ---------------------------------------------------------------------------

def categorise_gap_bridge(task: dict) -> list[str]:
    """Return ['GAP_BRIDGE'] if the task matches the gap-bridge rule."""
    return GAP_BRIDGE_CATEGORIES if detect_gap_bridge(task) else []
