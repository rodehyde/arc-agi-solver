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

CATEGORIES = (
    SIZE_CATEGORIES + CONTENT_CATEGORIES + GEOMETRIC_CATEGORIES
    + TRANSFORM_CATEGORIES + FLOOD_FILL_CATEGORIES + LOGICAL_OP_CATEGORIES
    + TILE_FILL_CATEGORIES + TILE_COMPRESS_CATEGORIES
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
    )


__all__ = ["categorise_task", "CATEGORIES"]
