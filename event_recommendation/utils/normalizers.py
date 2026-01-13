"""Normalization functions for text processing."""

import re
from typing import Optional


def normalize_event_type(s: Optional[str]) -> Optional[str]:
    """Upper-case and strip symbols like ® / ™ and extra punctuation."""
    if not s:
        return None
    s = s.strip().upper()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"\s+", " ", s)
    # Keep alphanumerics, spaces, '/', '&', '-', '+'
    s = re.sub(r"[^A-Z0-9 /&\-+]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_activity_heading(s: Optional[str]) -> Optional[str]:
    """Normalize activity heading for robust matching."""
    if not s:
        return None
    s = s.strip().upper()
    s = s.replace("®", "").replace("™", "")
    s = re.sub(r"\s+", " ", s)
    # Keep alphanumerics, spaces, '/', '&', '-', '+'
    s = re.sub(r"[^A-Z0-9 /&\-+]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_intensity(raw: Optional[str]) -> Optional[str]:
    """
    Map text like 'Low–Moderate', 'Level 2/3', etc. to low/moderate/high.
    Keep it intentionally simple.
    """
    if not raw:
        return None
    t = raw.strip().lower()
    if "low" in t or "gentle" in t:
        return "low"
    if "high" in t or "challenging" in t:
        return "high"
    if "moderate" in t or "medium" in t or "level 2" in t or "level 2/3" in t:
        return "moderate"
    # If it says "level 1" treat as low
    if "level 1" in t:
        return "low"
    # If it says "level 3" treat as high
    if "level 3" in t:
        return "high"
    return None


def normalize_age_focus(age_focus: Optional[str]) -> Optional[str]:
    """Normalize age focus to comma-separated string."""
    if not age_focus:
        return None
    from .extractors import extract_age_groups
    return ",".join(extract_age_groups(age_focus))


def normalize_city(city: Optional[str]) -> Optional[str]:
    """Normalize city name to lowercase."""
    if not city:
        return None
    return city.strip().lower()


def normalize_state(state: Optional[str]) -> Optional[str]:
    """Normalize state name to lowercase."""
    if not state:
        return None
    return state.strip().lower()

