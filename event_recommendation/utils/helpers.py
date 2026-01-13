"""Helper utility functions."""


def to_str_safe(v) -> str:
    """Safely convert value to string, handling lists."""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return str(v).strip() if v else ""

