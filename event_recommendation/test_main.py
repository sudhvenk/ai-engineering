"""Test script for main.py - validates structure and dependencies."""

import os
import sys

def test_path_resolution():
    """Test path resolution logic from main.py."""
    print("=" * 60)
    print("Testing Path Resolution")
    print("=" * 60)
    
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "helper":
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    documents_path = os.path.join(project_root, "documents")
    persist_dir = os.path.join(project_root, "rag_chroma")

    print(f"✓ Current directory: {current_dir}")
    print(f"✓ Project root: {project_root}")
    print(f"✓ Documents path: {documents_path}")
    print(f"✓ Persist dir: {persist_dir}")
    print(f"✓ Documents exist: {os.path.exists(documents_path)}")
    print(f"✓ Persist dir exists: {os.path.exists(persist_dir)}")
    
    if os.path.exists(documents_path):
        events_dir = os.path.join(documents_path, "Events")
        activity_dir = os.path.join(documents_path, "activityType")
        print(f"✓ Events directory exists: {os.path.exists(events_dir)}")
        print(f"✓ ActivityType directory exists: {os.path.exists(activity_dir)}")
    
    print()


def test_module_structure():
    """Test that all required modules exist."""
    print("=" * 60)
    print("Testing Module Structure")
    print("=" * 60)
    
    modules = [
        "rag",
        "rag/__init__.py",
        "rag/document_processing.py",
        "rag/retrieval.py",
        "rag/input_documents",
        "rag/input_documents/__init__.py",
        "rag/input_documents/loader.py",
        "vector_db",
        "vector_db/__init__.py",
        "vector_db/chroma_store.py",
        "chat_ui",
        "chat_ui/__init__.py",
        "chat_ui/chat_interface.py",
        "chat_ui/profile.py",
        "utils",
        "utils/__init__.py",
        "utils/normalizers.py",
        "utils/extractors.py",
        "utils/helpers.py",
        "main.py",
    ]
    
    all_exist = True
    for module in modules:
        exists = os.path.exists(module)
        status = "✓" if exists else "✗"
        print(f"{status} {module}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist


def test_imports():
    """Test imports (will fail if dependencies not installed)."""
    print("=" * 60)
    print("Testing Imports (requires dependencies)")
    print("=" * 60)
    
    sys.path.insert(0, '.')
    
    imports = [
        ("dotenv", "from dotenv import load_dotenv"),
        ("rag.input_documents.loader", "from rag.input_documents.loader import load_documents"),
        ("vector_db.chroma_store", "from vector_db.chroma_store import build_vectorstores"),
        ("chat_ui.chat_interface", "from chat_ui.chat_interface import launch_chat_interface"),
    ]
    
    results = {}
    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"✓ {name}")
            results[name] = True
        except Exception as e:
            error_msg = str(e)[:100]  # Truncate long error messages
            print(f"✗ {name}: {error_msg}")
            results[name] = False
    
    print()
    return results


def test_env_vars():
    """Test environment variable setup."""
    print("=" * 60)
    print("Testing Environment Variables")
    print("=" * 60)
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✓ .env file exists")
        
        # Try to load it
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            groq_key = os.getenv("GROQ_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            print(f"✓ GROQ_API_KEY set: {bool(groq_key)}")
            print(f"✓ OPENAI_API_KEY set: {bool(openai_key)}")
            
            if not groq_key:
                print("  ⚠ Warning: GROQ_API_KEY not found in .env")
            if not openai_key:
                print("  ⚠ Warning: OPENAI_API_KEY not found in .env")
        except ImportError:
            print("  ⚠ Cannot test env vars: dotenv not installed")
    else:
        print(f"✗ .env file not found")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MAIN.PY TEST SUITE")
    print("=" * 60 + "\n")
    
    # Test 1: Path resolution
    test_path_resolution()
    
    # Test 2: Module structure
    structure_ok = test_module_structure()
    
    # Test 3: Environment variables
    test_env_vars()
    
    # Test 4: Imports (may fail if deps not installed)
    import_results = test_imports()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Module structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    
    if import_results:
        passed = sum(1 for v in import_results.values() if v)
        total = len(import_results)
        print(f"Imports: {passed}/{total} passed")
        if passed < total:
            print("  ⚠ Some imports failed - install dependencies:")
            print("     pip install python-dotenv langchain langchain-openai langchain-chroma groq gradio")
    
    print("\n" + "=" * 60)
    print("To run main.py:")
    print("  1. Activate virtual environment: source activate/bin/activate")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print("  3. Set up .env file with GROQ_API_KEY and OPENAI_API_KEY")
    print("  4. Run: python main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

