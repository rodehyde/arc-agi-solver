"""
geometric_transforms.py — Programmatic detection of whole-grid geometric transforms.

A task passes if a single deterministic function maps *every* training input
exactly to its training output.  No descriptions, no embeddings — pure
pixel-level verification.

Transforms tried
----------------
D4 group (same-size output, 7 non-identity elements):
    rot90_cw       rotate 90° clockwise
    rot180         rotate 180°
    rot90_ccw      rotate 90° counter-clockwise
    flip_h         mirror left-right  (reflect across vertical axis)
    flip_v         mirror top-bottom  (reflect across horizontal axis)
    transpose      reflect across main diagonal  (swap rows and cols)
    anti_transpose reflect across anti-diagonal

Expansion by mirroring (output strictly larger than input):
    double_width_r    [g | flip_h(g)]           H × 2W
    double_width_l    [flip_h(g) | g]            H × 2W
    double_width_copy [g | g]                    H × 2W  (identity duplication)
    double_height_b   [g; flip_v(g)]             2H × W
    double_height_t   [flip_v(g); g]             2H × W
    double_height_copy[g; g]                     2H × W  (identity duplication)
    quad_hv           [[g, flip_h(g)],           2H × 2W
                       [flip_v(g), rot180(g)]]
    quad_vh           [[g, flip_v(g)],           2H × 2W
                       [flip_h(g), rot180(g)]]
    quad_rot          [[g, rot90_cw(g)],         2H × 2W
                       [rot90_ccw(g), rot180(g)]]
"""

import numpy as np

GEOMETRIC_CATEGORIES = ["GEOMETRIC_TRANSFORM"]

# ---------------------------------------------------------------------------
# Transform catalogue
# ---------------------------------------------------------------------------

_D4 = {
    "rot90_cw":       lambda g: np.rot90(g, k=3),
    "rot180":         lambda g: np.rot90(g, k=2),
    "rot90_ccw":      lambda g: np.rot90(g, k=1),
    "flip_h":         lambda g: g[:, ::-1],
    "flip_v":         lambda g: g[::-1, :],
    "transpose":      lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, k=2).T,
}

_EXPAND = {
    "double_width_r":    lambda g: np.concatenate([g, g[:, ::-1]],  axis=1),
    "double_width_l":    lambda g: np.concatenate([g[:, ::-1], g],  axis=1),
    "double_width_copy": lambda g: np.concatenate([g, g],            axis=1),
    "double_height_b":   lambda g: np.concatenate([g, g[::-1, :]],  axis=0),
    "double_height_t":   lambda g: np.concatenate([g[::-1, :], g],  axis=0),
    "double_height_copy":lambda g: np.concatenate([g, g],            axis=0),
    "quad_hv":  lambda g: np.block([[g,              g[:, ::-1]],
                                    [g[::-1, :],     g[::-1, ::-1]]]),
    "quad_vh":  lambda g: np.block([[g,              g[::-1, :]],
                                    [g[:, ::-1],     g[::-1, ::-1]]]),
    "quad_rot": lambda g: np.block([[g,              np.rot90(g, k=3)],
                                    [np.rot90(g, k=1), np.rot90(g, k=2)]]),
}

ALL_TRANSFORMS = {**_D4, **_EXPAND}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_transform(task: dict) -> tuple[str, object] | None:
    """
    Try every candidate transform against all training pairs.

    Returns (name, fn) for the first consistent transform, or None.
    A transform is *consistent* if it maps every training input exactly to
    the corresponding training output.
    """
    pairs = task["train"]
    for name, fn in ALL_TRANSFORMS.items():
        try:
            if all(
                np.array_equal(
                    fn(np.array(p["input"],  dtype=np.int32)),
                    np.array(p["output"], dtype=np.int32),
                )
                for p in pairs
            ):
                return name, fn
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Category interface (mirrors size_features / content_features pattern)
# ---------------------------------------------------------------------------

def categorise_geometric(task: dict) -> list[str]:
    """Return ['GEOMETRIC_TRANSFORM'] if a whole-grid transform is detected."""
    return GEOMETRIC_CATEGORIES if detect_transform(task) is not None else []
