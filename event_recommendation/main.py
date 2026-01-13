"""Main entry point for the activity recommendation chatbot."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag.input_documents.loader import load_documents
from vector_db.chroma_store import build_vectorstores, load_vectorstores
from chat_ui.chat_interface import launch_chat_interface


def main():
    """Main function to run the chatbot."""
    # Load environment variables
    load_dotenv()

    print("GROQ_API_KEY set:", bool(os.getenv("GROQ_API_KEY")))
    print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))

    # Get project root directory
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "helper":
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    documents_path = os.path.join(project_root, "documents")
    persist_dir = os.path.join(project_root, "rag_chroma")

    # Load documents
    print("Loading documents...")
    events_files, activity_files = load_documents(documents_path)

    # Build or load vector stores
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing vector stores...")
        stores = load_vectorstores(persist_dir=persist_dir)
    else:
        print("Building vector stores...")
        stores = build_vectorstores(
            events_files, activity_files, persist_dir=persist_dir
        )

    # Launch chat interface
    print("Launching chat interface...")
    launch_chat_interface(stores)


if __name__ == "__main__":
    main()

