from .size_features import categorise_size, SIZE_CATEGORIES
from .content_features import categorise_content, CONTENT_CATEGORIES
from .geometric_transforms import categorise_geometric, GEOMETRIC_CATEGORIES
from .transform_features import categorise_transform, detect_transform_detail, TRANSFORM_CATEGORIES
from .flood_fill import categorise_flood_fill, FLOOD_FILL_CATEGORIES
from .logical_ops import categorise_logical_op, LOGICAL_OP_CATEGORIES
from .tiling import (
    categorise_tile_fill, TILE_FILL_CATEGORIES,
    categorise_tile_compress, TILE_COMPRESS_CATEGORIES,
)
from .corner_staircase import categorise_corner_staircase, CORNER_STAIRCASE_CATEGORIES
from .rectangle_from_corners import (
    categorise_rectangle_from_corners, RECTANGLE_FROM_CORNERS_CATEGORIES,
)
from .gap_bridge import categorise_gap_bridge, GAP_BRIDGE_CATEGORIES
from .separator_grid_cross_fill import (
    categorise_separator_grid_cross_fill, SEPARATOR_GRID_CROSS_FILL_CATEGORIES,
)
from .bounding_box_fill import categorise_bounding_box_fill, BOUNDING_BOX_FILL_CATEGORIES
from .hole_fill_2x2 import categorise_hole_fill_2x2, HOLE_FILL_2X2_CATEGORIES
from .colour_marker_cross import categorise_colour_marker_cross, COLOUR_MARKER_CROSS_CATEGORIES
from .vertical_comb import categorise_vertical_comb, VERTICAL_COMB_CATEGORIES
from .separator_grid_diagonal_fill import (
    categorise_separator_grid_diagonal_fill, SEPARATOR_GRID_DIAGONAL_FILL_CATEGORIES,
)
from .border_encoded_scale import categorise_border_encoded_scale, BORDER_ENCODED_SCALE_CATEGORIES
from .quadrant_reflect import categorise_quadrant_reflect, QUADRANT_REFLECT_CATEGORIES
from .self_tile import categorise_self_tile, SELF_TILE_CATEGORIES
from .line_fill_by_colour import categorise_line_fill_by_colour, LINE_FILL_BY_COLOUR_CATEGORIES
from .row_fill_meet_middle import categorise_row_fill_meet_middle, ROW_FILL_MEET_MIDDLE_CATEGORIES
from .connect_aligned_pairs import (
    categorise_connect_aligned_pairs, CONNECT_ALIGNED_PAIRS_CATEGORIES,
)
from .quadrant_mirror import categorise_quadrant_mirror, QUADRANT_MIRROR_CATEGORIES
from .colour_remap import categorise_colour_remap, COLOUR_REMAP_CATEGORIES

CATEGORIES = (
    SIZE_CATEGORIES + CONTENT_CATEGORIES + GEOMETRIC_CATEGORIES
    + TRANSFORM_CATEGORIES + FLOOD_FILL_CATEGORIES + LOGICAL_OP_CATEGORIES
    + TILE_FILL_CATEGORIES + TILE_COMPRESS_CATEGORIES + CORNER_STAIRCASE_CATEGORIES
    + RECTANGLE_FROM_CORNERS_CATEGORIES + GAP_BRIDGE_CATEGORIES
    + SEPARATOR_GRID_CROSS_FILL_CATEGORIES + BOUNDING_BOX_FILL_CATEGORIES
    + HOLE_FILL_2X2_CATEGORIES + COLOUR_MARKER_CROSS_CATEGORIES
    + VERTICAL_COMB_CATEGORIES + SEPARATOR_GRID_DIAGONAL_FILL_CATEGORIES
    + BORDER_ENCODED_SCALE_CATEGORIES + QUADRANT_REFLECT_CATEGORIES
    + SELF_TILE_CATEGORIES + LINE_FILL_BY_COLOUR_CATEGORIES
    + ROW_FILL_MEET_MIDDLE_CATEGORIES
    + CONNECT_ALIGNED_PAIRS_CATEGORIES
    + QUADRANT_MIRROR_CATEGORIES
    + COLOUR_REMAP_CATEGORIES
)


def categorise_task(task: dict) -> list[str]:
    """Return all category labels that apply to this task."""
    return (
        categorise_size(task)
        + categorise_content(task)
        + categorise_geometric(task)
        + categorise_transform(task)
        + categorise_flood_fill(task)
        + categorise_logical_op(task)
        + categorise_tile_fill(task)
        + categorise_tile_compress(task)
        + categorise_corner_staircase(task)
        + categorise_rectangle_from_corners(task)
        + categorise_gap_bridge(task)
        + categorise_separator_grid_cross_fill(task)
        + categorise_bounding_box_fill(task)
        + categorise_hole_fill_2x2(task)
        + categorise_colour_marker_cross(task)
        + categorise_vertical_comb(task)
        + categorise_separator_grid_diagonal_fill(task)
        + categorise_border_encoded_scale(task)
        + categorise_quadrant_reflect(task)
        + categorise_self_tile(task)
        + categorise_line_fill_by_colour(task)
        + categorise_row_fill_meet_middle(task)
        + categorise_connect_aligned_pairs(task)
        + categorise_quadrant_mirror(task)
        + categorise_colour_remap(task)
    )


__all__ = ["categorise_task", "CATEGORIES"]
