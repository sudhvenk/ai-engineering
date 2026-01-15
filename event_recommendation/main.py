"""Main entry point for the activity recommendation chatbot."""

import os
import groq
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
    
    # Initialize Groq client and model
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = "openai/gpt-oss-120b"
    
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
    db_path = os.path.join(project_root, "events.db")
    reviews_db_path = os.path.join(project_root, "reviews.db")
    reviews_csv_path = os.path.join(project_root, "documents", "Reviews", "reviews_rag_2000.csv")

    # Load documents
    print("Loading documents...")
    events_files, activity_files = load_documents(documents_path)

    # Build or load vector stores and SQL database
    if os.path.exists(persist_dir) and os.listdir(persist_dir) and os.path.exists(db_path):
        print("Loading existing vector stores and database...")
        stores = load_vectorstores(
            persist_dir=persist_dir, 
            db_path=db_path,
            reviews_db_path=reviews_db_path
        )
    else:
        print("Building vector stores and database...")
        stores = build_vectorstores(
            events_files, 
            activity_files,
            groq_client=groq_client,
            model=model,
            persist_dir=persist_dir, 
            db_path=db_path,
            reviews_db_path=reviews_db_path
        )

    # Launch chat interface
    print("Launching chat interface...")
    launch_chat_interface(stores, groq_client=groq_client, model=model)


if __name__ == "__main__":
    main()

