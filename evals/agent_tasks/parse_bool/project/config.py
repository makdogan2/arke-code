"""Reading settings from environment-style strings."""


def parse_bool(value):
    """Accept true/false, yes/no, on/off and 1/0 in any case.
    Raise ValueError for anything else."""
    return bool(value)


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
