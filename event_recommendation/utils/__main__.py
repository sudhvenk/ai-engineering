"""Command-line interface for building reviews database."""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.build_reviews_db import build_reviews_database


def main():
    parser = argparse.ArgumentParser(
        description="Build reviews database from CSV file using LLM for metadata extraction"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="documents/Reviews/reviews_rag_2000.csv",
        help="Path to reviews CSV file (default: documents/Reviews/reviews_rag_2000.csv)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="reviews.db",
        help="Path to reviews SQLite database (default: reviews.db)"
    )
    parser.add_argument(
        "--client",
        type=str,
        choices=["groq", "ollama"],
        default="groq",
        help="LLM client to use: 'groq' or 'ollama' (default: groq)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: 'openai/gpt-oss-120b' for Groq, 'llama3.2:latest' for Ollama)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use regex-based extraction instead of LLM (faster but less accurate)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of reviews to process in a single LLM call (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Determine model if not provided
    if args.model is None:
        if args.client == "groq":
            args.model = "openai/gpt-oss-120b"
        else:
            args.model = "llama3.2:latest"
    
    # Initialize client
    if args.client == "groq":
        import groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not found in environment variables.")
            print("Please set it in your .env file or environment.")
            sys.exit(1)
        client = groq.Groq(api_key=api_key)
        print(f"Using Groq client with model: {args.model}")
    else:  # ollama
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        print(f"Using Ollama client with model: {args.model}")
        print("Make sure Ollama server is running at http://localhost:11434")
    
    # Build reviews database
    try:
        reviews_db = build_reviews_database(
            reviews_csv_path=args.csv_path,
            reviews_db_path=args.db_path,
            llm_client=client if not args.no_llm else None,
            model=args.model,
            use_llm=not args.no_llm,
            batch_size=args.batch_size,
        )
        
        review_count = reviews_db.count_reviews()
        print(f"\n✓ Successfully built reviews database!")
        print(f"  Database: {args.db_path}")
        print(f"  Total reviews: {review_count}")
        
    except Exception as e:
        print(f"\n✗ Error building reviews database: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
