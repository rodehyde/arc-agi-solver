"""
RECOLOR_BY_HOLES (0a2355a6 evaluation)

Input: grid of 0s with several connected shapes all colored 8.

Rule: recolor each shape based on the number of enclosed holes
(connected zero regions fully surrounded by the shape):
  1 hole  -> color 1
  2 holes -> color 3   (2 and 3 are swapped relative to counting)
  3 holes -> color 2
  n holes -> color n   (for n >= 4)

Detection: all non-zero cells are color 8; hole-count recoloring
matches all training pairs.
"""

from collections import deque


def _get_shapes(grid):
    H, W = len(grid), len(grid[0])
    vis = [[False] * W for _ in range(H)]
    shapes = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not vis[r][c]:
                color = grid[r][c]
                cells = []
                q = deque([(r, c)])
                vis[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not vis[nr][nc] and grid[nr][nc] == color):
                            vis[nr][nc] = True
                            q.append((nr, nc))
                shapes.append((color, cells))
    return shapes


def _count_holes(cells, grid):
    cell_set = set(cells)
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)

    exterior = set()

    def flood(sr, sc):
        if (sr, sc) in exterior or (sr, sc) in cell_set:
            return
        if not (r1 <= sr <= r2 and c1 <= sc <= c2):
            return
        q = deque([(sr, sc)])
        exterior.add((sr, sc))
        while q:
            cr, cc = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (r1 <= nr <= r2 and c1 <= nc <= c2
                        and (nr, nc) not in exterior
                        and (nr, nc) not in cell_set):
                    exterior.add((nr, nc))
                    q.append((nr, nc))

    for r in range(r1, r2 + 1):
        for c in [c1, c2]:
            if (r, c) not in cell_set:
                flood(r, c)
    for c in range(c1, c2 + 1):
        for r in [r1, r2]:
            if (r, c) not in cell_set:
                flood(r, c)

    interior = {
        (r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
        if (r, c) not in cell_set and (r, c) not in exterior
    }

    vis2 = set()
    n_holes = 0
    for z in interior:
        if z not in vis2:
            n_holes += 1
            q = deque([z])
            vis2.add(z)
            while q:
                cr, cc = q.popleft()
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nz = (cr + dr, cc + dc)
                    if nz in interior and nz not in vis2:
                        vis2.add(nz)
                        q.append(nz)
    return n_holes


def _holes_to_color(n):
    # 2 and 3 are swapped; all others map to themselves
    return n ^ 1 if n in (2, 3) else n


def _predict(inp):
    H, W = len(inp), len(inp[0])
    shapes = _get_shapes(inp)
    if not shapes:
        return None
    out = [[0] * W for _ in range(H)]
    for color, cells in shapes:
        n = _count_holes(cells, inp)
        new_color = _holes_to_color(n)
        for r, c in cells:
            out[r][c] = new_color
    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        predicted = _predict(inp)
        if predicted is None or predicted != out:
            return False
    return True


def solve(inp):
    result = _predict(inp)
    return result if result is not None else [list(row) for row in inp]


def categorise(task):
    return detect(task)
