"""Pagination helpers for list views."""


def paginate(items, page, per_page):
    """Return the items on a 1-based page (an empty list past the end)."""
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be positive")
    start = page * per_page
    return items[start:start + per_page]


def page_count(n_items, per_page):
    """How many pages are needed for n_items."""
    if per_page < 1:
        raise ValueError("per_page must be positive")
    return -(-n_items // per_page)
