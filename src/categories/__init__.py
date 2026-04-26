from .size_features import categorise_size, SIZE_CATEGORIES
from .content_features import categorise_content, CONTENT_CATEGORIES
from .geometric_transforms import categorise_geometric, GEOMETRIC_CATEGORIES
from .transform_features import categorise_transform, detect_transform_detail, TRANSFORM_CATEGORIES

CATEGORIES = SIZE_CATEGORIES + CONTENT_CATEGORIES + GEOMETRIC_CATEGORIES + TRANSFORM_CATEGORIES


def categorise_task(task: dict) -> list[str]:
    """Return all category labels that apply to this task."""
    return (
        categorise_size(task)
        + categorise_content(task)
        + categorise_geometric(task)
        + categorise_transform(task)
    )


__all__ = ["categorise_task", "CATEGORIES"]
