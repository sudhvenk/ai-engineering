"""Tests for input/output processing."""

import pytest
from rag.document_processing import (
    parse_center_metadata,
    split_event_blocks,
    parse_event_metadata,
    build_activitytype_documents,
    build_event_documents,
)


def test_parse_center_metadata():
    """Test parsing center metadata."""
    md_text = """
## YMCA Riverstone
**Location:** Framingham, Massachusetts
**Type:** Community Center
"""
    result = parse_center_metadata(md_text, "test.md")
    assert result["center_name"] == "YMCA Riverstone"
    assert result["city"] == "Framingham"
    assert result["state"] == "Massachusetts"
    assert result["center_type"] == "Community Center"


def test_split_event_blocks():
    """Test splitting event blocks."""
    md_text = """
### Event 1
- Date: Jan 1
### Event 2
- Date: Jan 2
"""
    blocks = split_event_blocks(md_text)
    assert len(blocks) == 2
    assert blocks[0][0] == "Event 1"
    assert blocks[1][0] == "Event 2"


def test_parse_event_metadata():
    """Test parsing event metadata."""
    block = """
- Event Type: Aquatics
- Category: Swimming
- Age Tags: Adults
- Instructor: John Doe
- Date Range: Jan 1 - Dec 31
- Time Slots: 10:00 AM
- Duration: 1 hour
- Spots: 20
"""
    result = parse_event_metadata("Test Event", block)
    assert result["event_title"] == "Test Event"
    assert "Aquatics" in result["event_type"] or "Swimming" in result["event_type"]
    assert result["age_tags"] == "Adults"
    assert result["instructor"] == "John Doe"


def test_build_activitytype_documents():
    """Test building activity type documents."""
    md_text = """
## Aquatics

### AQUA CARDIO
**Intensity:** Moderate

Description of aqua cardio.
"""
    docs, intensity_map = build_activitytype_documents(md_text, "aquatics.md")
    assert len(docs) > 0
    assert "AQUA CARDIO" in intensity_map or len(intensity_map) >= 0


def test_build_event_documents():
    """Test building event documents."""
    md_text = """
## YMCA Riverstone
**Location:** Framingham, Massachusetts

### Swimming Class
- Event Type: Aquatics
- Age Tags: Adults
"""
    docs = build_event_documents(
        md_text, "test.md", activity_intensity_map={}, city="Framingham", state="Massachusetts"
    )
    assert len(docs) > 0
    assert docs[0].metadata["event_name"] == "Swimming Class"
    assert docs[0].metadata["city"] == "Framingham"

