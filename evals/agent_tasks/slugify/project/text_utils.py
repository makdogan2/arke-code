"""Text helpers for building URLs."""
import re


def slugify(title):
    """Turn a title into a URL slug: 'Hello, World!' -> 'hello-world'.

    Lowercase ASCII letters and digits are kept; every run of other
    characters becomes a single dash; no dash at either end.
    """
    text = title.lower()
    text = re.sub(r"[^a-z0-9]", "-", text)
    return text.strip("-")


def truncate(text, limit):
    """Cut text to at most `limit` characters, ending with '...' if cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
