"""Simple tests that don't require external dependencies."""

import sys
sys.path.insert(0, '.')


def test_build_chroma_where():
    """Test building Chroma where clause - import function directly."""
    # Import the function directly to avoid langchain dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chroma_store", "vector_db/chroma_store.py"
    )
    chroma_store = importlib.util.module_from_spec(spec)
    
    # Mock the langchain imports
    import types
    mock_module = types.ModuleType('langchain_openai')
    mock_module.OpenAIEmbeddings = type('Mock', (), {})
    sys.modules['langchain_openai'] = mock_module
    
    mock_chroma = types.ModuleType('langchain_chroma')
    mock_chroma.Chroma = type('Mock', (), {})
    sys.modules['langchain_chroma'] = mock_chroma
    
    mock_doc = types.ModuleType('langchain_core.documents')
    mock_doc.Document = type('Mock', (), {})
    sys.modules['langchain_core'] = types.ModuleType('langchain_core')
    sys.modules['langchain_core.documents'] = mock_doc
    
    # Now import the function
    spec.loader.exec_module(chroma_store)
    build_chroma_where = chroma_store.build_chroma_where
    
    # Single filter
    result = build_chroma_where({"city": "boston"})
    assert result == {"city": "boston"}, f"Expected {{'city': 'boston'}}, got {result}"
    print("✓ test_build_chroma_where_single_filter passed")
    
    # Multiple filters
    result = build_chroma_where({"city": "boston", "age_contains": "adults"})
    assert "$and" in result, f"Expected $and in result, got {result}"
    assert len(result["$and"]) == 2, f"Expected 2 clauses, got {len(result['$and'])}"
    print("✓ test_build_chroma_where_multiple_filters passed")
    
    # Empty filter
    result = build_chroma_where({})
    assert result is None, f"Expected None, got {result}"
    print("✓ test_build_chroma_where_empty passed")
    
    # Empty string filter
    result = build_chroma_where({"city": "boston", "state": ""})
    if "$and" in result:
        assert len(result["$and"]) == 1, "Should filter out empty strings"
    else:
        assert "city" in result, "Should only include city"
    print("✓ test_build_chroma_where_empty_string passed")


def test_normalizers():
    """Test normalization functions."""
    from utils.normalizers import (
        normalize_city,
        normalize_state,
        normalize_event_type,
        normalize_activity_heading,
        normalize_intensity,
    )
    
    assert normalize_city("Boston") == "boston"
    print("✓ normalize_city passed")
    
    assert normalize_state("Massachusetts") == "massachusetts"
    print("✓ normalize_state passed")
    
    assert normalize_event_type("Aqua Zumba®") == "AQUA ZUMBA"
    print("✓ normalize_event_type passed")
    
    assert normalize_activity_heading("Aqua Cardio") == "AQUA CARDIO"
    print("✓ normalize_activity_heading passed")
    
    assert normalize_intensity("Low") == "low"
    assert normalize_intensity("High intensity") == "high"
    assert normalize_intensity("Moderate") == "moderate"
    print("✓ normalize_intensity passed")


def test_extractors():
    """Test extraction functions."""
    from utils.extractors import extract_age_range, extract_age_groups
    
    assert extract_age_range("Ages: 6-10") == (6, 10)
    assert extract_age_range("Age 8-12") == (8, 12)
    assert extract_age_range("Ages: 18+") == (18, None)
    print("✓ extract_age_range passed")
    
    groups = extract_age_groups("Ages: 6-10")
    assert "kids" in groups
    print("✓ extract_age_groups passed")


def test_helpers():
    """Test helper functions."""
    from utils.helpers import to_str_safe
    
    assert to_str_safe("test") == "test"
    assert to_str_safe(["a", "b"]) == "a, b"
    assert to_str_safe(None) == ""
    print("✓ to_str_safe passed")


def test_document_processing():
    """Test document processing functions."""
    from rag.document_processing import parse_center_metadata, split_event_blocks
    
    md_text = """
## YMCA Riverstone
**Location:** Framingham, Massachusetts
**Type:** Community Center
"""
    result = parse_center_metadata(md_text, "test.md")
    assert result["center_name"] == "YMCA Riverstone"
    assert result["city"] == "Framingham"
    assert result["state"] == "Massachusetts"
    print("✓ parse_center_metadata passed")
    
    md_text = """
### Event 1
- Date: Jan 1
### Event 2
- Date: Jan 2
"""
    blocks = split_event_blocks(md_text)
    assert len(blocks) == 2
    assert blocks[0][0] == "Event 1"
    print("✓ split_event_blocks passed")


if __name__ == "__main__":
    print("Running simple tests (no external dependencies required)...\n")
    
    try:
        test_build_chroma_where()
        print()
        test_normalizers()
        print()
        test_extractors()
        print()
        test_helpers()
        print()
        test_document_processing()
        print()
        print("=" * 50)
        print("All simple tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
