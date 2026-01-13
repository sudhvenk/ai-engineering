"""Document processing and parsing for RAG."""

import re
import os
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from utils.normalizers import (
    normalize_event_type,
    normalize_activity_heading,
    normalize_intensity,
)
from utils.extractors import extract_age_range, extract_age_groups, infer_intensity_from_text


# Regex patterns for parsing
EVENT_HEADING_RE = re.compile(r"^###\s+(.*)\s*$", re.MULTILINE)
FIELD_RE = {
    "event_type": re.compile(r"^\s*-\s*Event Type:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "category": re.compile(r"^\s*-\s*Category:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "age_tags": re.compile(r"^\s*-\s*Age Tags:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "instructor": re.compile(r"^\s*-\s*(Instructor|Facilitator):\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "date_range": re.compile(r"^\s*-\s*Date Range:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "time_slots": re.compile(r"^\s*-\s*Time Slots:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "duration": re.compile(r"^\s*-\s*Duration:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
    "spots": re.compile(r"^\s*-\s*Spots:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE),
}

CENTER_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
LOCATION_RE = re.compile(r"^\*\*Location:\*\*\s*(.+?)\s*$", re.MULTILINE)
TYPE_RE = re.compile(r"^\*\*Type:\*\*\s*(.+?)\s*$", re.MULTILINE)
PAGE_RE = re.compile(r"^#\s+PAGE\s+\d+\s+—\s+(.+?)\s*$", re.MULTILINE)
_EVENT_BLOCK_RE = re.compile(r"(?m)^###\s+(.+?)\s*$")
_INTENSITY_RE = re.compile(r"(?mi)^\s*\*\*Intensity:\*\*\s*(.+?)\s*$")


def _safe_find(regex: re.Pattern, text: str) -> Optional[str]:
    """Safely find first match in text."""
    m = regex.search(text)
    return m.group(1).strip() if m else None


def _safe_find2(regex: re.Pattern, text: str) -> Optional[str]:
    """Safely find second group in match."""
    m = regex.search(text)
    return m.group(2).strip() if m else None


def parse_center_metadata(md_text: str, source: str) -> Dict[str, Optional[str]]:
    """Parse center metadata from markdown text."""
    center_name = _safe_find(CENTER_RE, md_text)
    location = _safe_find(LOCATION_RE, md_text)
    center_type = _safe_find(TYPE_RE, md_text)

    city, state = None, None
    if location:
        # "Salem, Massachusetts" or "Plymouth, Massachusetts"
        parts = [p.strip() for p in location.split(",")]
        if len(parts) >= 2:
            city, state = parts[0], parts[1]

    return {
        "source": source,
        "center_name": center_name,
        "center_type": center_type,
        "city": city,
        "state": state,
    }


def split_event_blocks(md_text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (event_title, event_block_text).
    Event blocks start with '### ' and continue until next '### ' or end.
    """
    matches = list(EVENT_HEADING_RE.finditer(md_text))
    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        block = md_text[start:end].strip()
        blocks.append((title, block))
    return blocks


def parse_event_metadata(event_title: str, block: str) -> Dict[str, Optional[str]]:
    """Parse event metadata from event block."""
    # Event Type may appear as "Event Type:" or "Category:" depending on file style
    event_type = _safe_find(FIELD_RE["event_type"], block) or _safe_find(FIELD_RE["category"], block)

    age_tags = _safe_find(FIELD_RE["age_tags"], block)
    instructor = _safe_find2(FIELD_RE["instructor"], block)
    date_range = _safe_find(FIELD_RE["date_range"], block)
    time_slots = _safe_find(FIELD_RE["time_slots"], block)
    duration = _safe_find(FIELD_RE["duration"], block)
    spots = _safe_find(FIELD_RE["spots"], block)

    return {
        "event_title": event_title,
        "event_type": event_type,
        "age_tags": age_tags,
        "instructor": instructor,
        "date_range": date_range,
        "time_slots": time_slots,
        "duration": duration,
        "spots": spots,
    }


def build_activitytype_documents(
    md_text: str,
    source: str,
) -> Tuple[List[Document], Dict[str, str]]:
    """
    Build activity type documents from markdown.
    
    Returns:
        - activity_docs: Documents with metadata
        - intensity_map: dict[activity_heading_norm] -> 'low'|'moderate'|'high'
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks = splitter.split_text(md_text)

    docs: List[Document] = []
    intensity_map: Dict[str, str] = {}

    for c in chunks:
        meta = c.metadata or {}
        # Split heading from source file name, strip .md
        heading = source.split(".")[0].strip()
        heading_norm = normalize_activity_heading(
            (meta.get("h3") or meta.get("h2") or heading).strip()
        )

        # Skip chunks that aren't actual activity headings
        if not heading_norm:
            continue
        if heading_norm.startswith("PAGE "):  # safety if a brochure sneaks in
            continue

        # Parse intensity line if present
        m = _INTENSITY_RE.search(c.page_content)
        intensity = normalize_intensity(m.group(1)) if m else None

        if intensity and heading_norm not in intensity_map:
            intensity_map[heading_norm] = intensity

        docs.append(
            Document(
                page_content=c.page_content.strip(),
                metadata={
                    "source": source,
                    "activity_heading": heading,
                    "activity_heading_norm": heading_norm,
                    "intensity": intensity,  # may be None if line not present
                },
            )
        )

    return docs, intensity_map


def build_event_documents(
    md_text: str,
    source: str,
    activity_intensity_map: Optional[Dict[str, str]] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
) -> List[Document]:
    """
    Split a brochure into one Document per event (each '### ...' block).

    Metadata:
    - event_name
    - event_type (normalized)
    - age_min, age_max
    - age_contains (bucketed groups)
    - city, state if present
    - intensity (low/moderate/high) [prefer from activity_intensity_map]
    """
    matches = list(_EVENT_BLOCK_RE.finditer(md_text))
    if not matches:
        return []

    docs: List[Document] = []

    for i, m in enumerate(matches):
        event_name = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        block = md_text[start:end].strip()

        # Extract Event Type
        event_type_raw = None
        mt = re.search(r"(?mi)^\s*-\s*Event Type:\s*(.+?)\s*$", block)
        if mt:
            event_type_raw = mt.group(1).strip()

        event_type = normalize_event_type(event_type_raw)

        # Ages numeric + buckets
        age_min, age_max = extract_age_range(block)
        age_contains_list = extract_age_groups(block)
        # Convert list to string for ChromaDB (which doesn't support list metadata)
        age_contains = ", ".join(age_contains_list) if age_contains_list else None

        # Intensity: prefer map from activityType docs (keyed by normalized event_type)
        intensity = None
        if activity_intensity_map and event_type:
            intensity = activity_intensity_map.get(event_type)
        if not intensity:
            intensity = infer_intensity_from_text(block)

        docs.append(
            Document(
                page_content=block,
                metadata={
                    "source": source,
                    "event_name": event_name,
                    "event_type": event_type,  # normalized (e.g., "AQUA ZUMBA")
                    "event_type_raw": event_type_raw,  # optional debugging
                    "age_min": age_min,
                    "age_max": age_max,
                    "city": city,
                    "state": state,
                    "age_contains": age_contains,  # comma-separated string
                    "intensity": intensity,  # low/moderate/high or None
                },
            )
        )

    return docs

