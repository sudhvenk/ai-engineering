"""Extraction functions for parsing text data."""

import re
from typing import List, Optional, Tuple

# Age keywords for classification
AGE_KEYWORDS = {
    "kids": ["kids", "children", "youth"],
    "teens": ["teen", "teens"],
    "young_adults": ["young adult", "college"],
    "adults": ["adult", "adults"],
    "seniors": ["senior", "older adult", "55+", "60+", "65+"],
    "all": ["all ages", "family"],
}


def _bucket_from_age_range(
    min_age: Optional[int], max_age: Optional[int]
) -> List[str]:
    """
    Convert numeric min/max into broad groups.
    Rules are intentionally simple and deterministic.
    """
    if min_age is None and max_age is None:
        return []

    # If only one side exists, treat it as both ends
    if min_age is None:
        min_age = max_age
    if max_age is None:
        max_age = min_age

    groups = set()

    # kids: <= 12
    if min_age <= 12:
        groups.add("kids")

    # teens: 13-17
    if max_age >= 13 and min_age <= 17:
        groups.add("teens")

    # young adults: 18-25
    if max_age >= 18 and min_age <= 25:
        groups.add("young_adults")

    # adults: 26-59 (or any adult mention)
    if max_age >= 26 and min_age <= 59:
        groups.add("adults")

    # seniors: 60+
    if max_age >= 60:
        groups.add("seniors")

    return sorted(groups)


def extract_age_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts numeric age range from patterns like:
      - "Ages: 6–10"
      - "Age 8-12"
      - "Ages 16 - 18"
      - "Ages: 18+"
      - "55+"
    Returns (min_age, max_age) where max_age can be None for open-ended.
    """
    t = text

    # Ages: 6–10 / 6-10 / 6 — 10
    m = re.search(r"(?i)\bages?\s*[:–-]?\s*(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Age: 8-12
    m = re.search(r"(?i)\bage\s*[:–-]?\s*(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", t)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Ages: 18+
    m = re.search(r"(?i)\bages?\s*[:–-]?\s*(\d{1,2})\s*\+\b", t)
    if m:
        return int(m.group(1)), None

    # Standalone 55+ / 60+
    m = re.search(r"(?i)\b(\d{2})\s*\+\b", t)
    if m:
        return int(m.group(1)), None

    return None, None


def extract_age_groups(text: str) -> List[str]:
    """
    Prefer numeric extraction if present; fall back to keyword detection.
    Default to adults if nothing found.
    """
    text_l = text.lower()

    min_age, max_age = extract_age_range(text)
    groups = set(_bucket_from_age_range(min_age, max_age))

    # keyword-based additions
    for g, kws in AGE_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            groups.add(g)

    # If "all" appears, keep it and optionally drop others (your choice)
    if "all" in groups:
        return ["all"]

    if not groups:
        groups.add("adults")

    return sorted(groups)


def infer_intensity_from_text(text: str) -> Optional[str]:
    """
    Heuristic fallback if event block itself contains intensity cues.
    Prefer using activity-type definitions instead (recommended).
    """
    t = text.lower()
    # Strong cues first
    if any(
        x in t
        for x in [
            "low impact",
            "gentle",
            "restorative",
            "beginner",
            "chair",
            "arthritis",
        ]
    ):
        return "low"
    if any(
        x in t
        for x in [
            "high intensity",
            "interval",
            "boot camp",
            "fast-paced",
            "challenging",
        ]
    ):
        return "high"
    if any(
        x in t
        for x in ["moderate", "all levels", "level 2", "level 2/3"]
    ):
        return "moderate"
    return None

