from .size_features import categorise_size, SIZE_CATEGORIES
from .content_features import categorise_content, CONTENT_CATEGORIES

CATEGORIES = SIZE_CATEGORIES + CONTENT_CATEGORIES


def categorise_task(task: dict) -> list[str]:
    """Return all category labels that apply to this task."""
    return categorise_size(task) + categorise_content(task)


__all__ = ["categorise_task", "CATEGORIES"]
