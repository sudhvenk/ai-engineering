"""Load documents from filesystem."""

import os
import glob
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(documents_path: str) -> tuple[List[str], List[str]]:
    """
    Load markdown documents from the documents directory.
    
    Args:
        documents_path: Path to the documents directory
        
    Returns:
        Tuple of (events_md_files, activitytype_md_files) - lists of file paths
    """
    # Create list of events md file list
    events_md_files: List[str] = []
    # Create list of activity type md file list
    activitytype_md_files: List[str] = []

    folders = glob.glob(os.path.join(documents_path, "*"))

    for folder in folders:
        if os.path.isdir(folder):
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            )
            folder_docs = loader.load()
            print(f"Loaded {len(folder_docs)} documents from {doc_type}")
            for doc in folder_docs:
                # Load events in events_md_files list (append file path, not Document object)
                if doc_type == "Events":
                    events_md_files.append(doc.metadata.get("source", ""))
                # Load activity types in activitytype_md_files list (append file path, not Document object)
                elif doc_type == "activityType":
                    activitytype_md_files.append(doc.metadata.get("source", ""))
                doc.metadata["doc_type"] = doc_type

    return events_md_files, activitytype_md_files

