"""
COLOR_MATCH_BY_SHAPE (2a5f8217 evaluation)

Input: grid with several connected components in various non-zero colors,
plus some components in color 1.

Rule: each color-1 component has the same geometric shape (normalized
cell offsets) as exactly one non-1 colored component. Recolor each color-1
component to the matching non-1 component's color. All other cells
are unchanged.

Detection: every color-1 component has a unique shape match among non-1
components; prediction matches all training pairs.
"""


def _normalize(cells):
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _find_components(grid):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    components = []
    for sr in range(H):
        for sc in range(W):
            v = grid[sr][sc]
            if v != 0 and not visited[sr][sc]:
                cells = []
                stack = [(sr, sc)]
                visited[sr][sc] = True
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((v, cells))
    return components


def _predict(inp):
    components = _find_components(inp)
    color1_comps = [(cells, _normalize(cells)) for v, cells in components if v == 1]
    non1_comps = [(v, _normalize(cells)) for v, cells in components if v != 1]
    shape_to_color = {shape: v for v, shape in non1_comps}

    H, W = len(inp), len(inp[0])
    out = [list(row) for row in inp]
    for cells, shape in color1_comps:
        target = shape_to_color.get(shape)
        if target is not None:
            for r, c in cells:
                out[r][c] = target
    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if _predict(inp) != out:
            return False
    return True


def solve(inp):
    return _predict(inp)


def categorise(task):
    return detect(task)
