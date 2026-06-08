"""
CONCENTRIC_LAYERS (516b51b7 evaluation)

Input: grid of 0s containing one or more solid rectangular blocks of 1s.

Rule: for each block, color each cell by its concentric depth (distance
from the nearest edge + 1). The color assignment follows a triangle wave
capped at 3:
    depth = min(local_r, BH-1-local_r, local_c, BW-1-local_c) + 1
    color = 3 - abs(depth - 3)
This gives: depth 1→1, 2→2, 3→3, 4→2, 5→1, 6→2, ...

Detection: all input non-zero cells are 1; prediction matches output
across all training pairs.
"""


def _predict(inp):
    H, W = len(inp), len(inp[0])
    out = [list(row) for row in inp]
    visited = [[False] * W for _ in range(H)]

    for sr in range(H):
        for sc in range(W):
            if inp[sr][sc] == 1 and not visited[sr][sc]:
                cells = []
                stack = [(sr, sc)]
                visited[sr][sc] = True
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                r1 = min(r for r, c in cells)
                r2 = max(r for r, c in cells)
                c1 = min(c for r, c in cells)
                c2 = max(c for r, c in cells)
                bh = r2 - r1 + 1
                bw = c2 - c1 + 1

                for r, c in cells:
                    lr, lc = r - r1, c - c1
                    depth = min(lr, bh - 1 - lr, lc, bw - 1 - lc) + 1
                    out[r][c] = 3 - abs(depth - 3)

    return out


def detect(task):
    pairs = task["train"]
    if not pairs:
        return False
    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        if any(inp[r][c] not in (0, 1) for r in range(len(inp)) for c in range(len(inp[0]))):
            return False
        if _predict(inp) != out:
            return False
    return True


def solve(inp):
    return _predict(inp)


def categorise(task):
    return detect(task)
