"""
EXTRACT_LR_SYMMETRIC (d56f2372 eval)

Multiple coloured shapes appear in the input; exactly one is left-right
symmetric. Output = that shape, cropped to its bounding box.
"""

from collections import defaultdict


def _bfs4(grid, sr, sc):
    rows, cols = len(grid), len(grid[0])
    val = grid[sr][sc]
    visited = {(sr, sc)}
    queue = [(sr, sc)]
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == val:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


def _crop(cells, colour):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    g = [[0] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        g[r - r0][c - c0] = colour
    return g


def _is_lr_symmetric(grid):
    return all(row == list(reversed(row)) for row in grid)


def detect(task):
    pairs = task["train"]
    if len(pairs) < 2:
        return False
    for pair in pairs:
        inp = pair["input"]
        rows, cols = len(inp), len(inp[0])
        seen = set()
        by_colour = defaultdict(set)
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != 0 and (r, c) not in seen:
                    cells = _bfs4(inp, r, c)
                    seen |= cells
                    by_colour[inp[r][c]] |= cells
        symmetric = [col for col, cells in by_colour.items() if _is_lr_symmetric(_crop(cells, col))]
        if len(symmetric) != 1:
            return False
    return True


def solve(inp):
    rows, cols = len(inp), len(inp[0])
    seen = set()
    by_colour = defaultdict(set)
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] != 0 and (r, c) not in seen:
                cells = _bfs4(inp, r, c)
                seen |= cells
                by_colour[inp[r][c]] |= cells
    for colour, cells in sorted(by_colour.items()):
        g = _crop(cells, colour)
        if _is_lr_symmetric(g):
            return g
    return None


def categorise(task):
    return detect(task)
