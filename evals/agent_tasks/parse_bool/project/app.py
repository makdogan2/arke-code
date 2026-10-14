"""Application settings built from an environment dict."""
from config import parse_bool, parse_int


def load_settings(env):
    return {
        "debug": parse_bool(env.get("DEBUG", "false")),
        "workers": parse_int(env.get("WORKERS"), default=2),
    }
