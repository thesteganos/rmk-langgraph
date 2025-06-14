import os
from langchain_community.document_loaders import PyPDFLoader # Changed from PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Removed HuggingFaceEmbeddings import, will get from src.utils
from langchain_chroma import Chroma
from src.utils import get_embedding_model # Added import for centralized model loading

DATA_PATH = "data"
DB_PATH = "db"
PROCESSED_LOG_FILE = os.path.join(DB_PATH, "processed_files.log")

import sys # Added for stderr

def get_processed_files():
    """Reads the log of already processed files."""
    try:
        with open(PROCESSED_LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def log_processed_file(filename):
    """Adds a new file to the processed log."""
    try:
        with open(PROCESSED_LOG_FILE, "a") as f:
            f.write(filename + "\n")
    except IOError as e:
        print(f"Error: Could not write to processed log file {PROCESSED_LOG_FILE}: {e}", file=sys.stderr)

def main():
    """Performs incremental ingestion of new PDF files into the ChromaDB vector store."""
    print("--- Starting Incremental Ingestion Process ---")
    os.makedirs(DB_PATH, exist_ok=True)

    try:
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            print(f"The '{DATA_PATH}' directory is empty or does not exist. Please add PDF files to ingest.")
            return
    except OSError as e:
        print(f"Error: Could not access data directory {DATA_PATH}: {e}", file=sys.stderr)
        return

    processed_files = get_processed_files()
    try:
        all_files = set(os.listdir(DATA_PATH))
    except OSError as e:
        print(f"Error: Could not list files in data directory {DATA_PATH}: {e}", file=sys.stderr)
        return
    new_files = sorted(list(all_files - processed_files))

    if not new_files:
        print("No new documents to process. Knowledge base is up-to-date.")
        return

    print(f"Found {len(new_files)} new document(s) to process: {new_files}")

    all_new_documents = []
    successfully_loaded_files = [] # Keep track of files that loaded successfully

    print("Loading new documents one by one...")
    for filename in new_files:
        file_path = os.path.join(DATA_PATH, filename)
        try:
            print(f"Processing '{file_path}'...")
            loader = PyPDFLoader(file_path)
            documents_from_file = loader.load() # Returns a list of Document objects
            if documents_from_file:
                all_new_documents.extend(documents_from_file)
                successfully_loaded_files.append(filename) # Add to list of successfully loaded files
                print(f"Successfully loaded {len(documents_from_file)} document(s) from '{filename}'.")
            else:
                print(f"Warning: No documents found in '{filename}'. It might be empty or corrupted.", file=sys.stderr)
        except Exception as e: # Catching a broad exception as PyPDFLoader can raise various issues
            print(f"Error: Could not load or process PDF file {file_path}: {e}", file=sys.stderr)
            print(f"Skipping file {file_path} due to error.", file=sys.stderr)

    if not all_new_documents:
        print("No documents were successfully loaded from the new files. Nothing to add to the knowledge base.")
        # No need to log processed files if none were successful
        print("-----------------------------------------")
        print("Incremental ingestion complete (no new data added).")
        print("-----------------------------------------")
        return

    print(f"Total of {len(all_new_documents)} new document(s) loaded from {len(successfully_loaded_files)} file(s).")

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_new_documents) # Use all_new_documents here
        print(f"Split into {len(texts)} chunks.")

        print("Loading embedding model...")
        embeddings = get_embedding_model()
        if not embeddings:
            # The error is already printed by get_embedding_model,
            # but we should not proceed if the model failed to load.
            print("Halting ingestion due to embedding model loading failure.", file=sys.stderr)
            return # Exit main()
        print("Embeddings model loaded successfully.")

        print("Adding new documents to the vector store...")
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        db.add_documents(texts) # Add the processed texts
        print("Knowledge base updated successfully.")

        # Log only the successfully processed files
        for filename in successfully_loaded_files:
            log_processed_file(filename)
        print(f"Updated the processed files log for {len(successfully_loaded_files)} file(s).")

    except Exception as e:
        # This is a general catch-all for errors during splitting, embedding, or DB add.
        # If this block is hit, none of the files from this run should be logged as processed.
        print(f"An error occurred during text processing or database update: {e}", file=sys.stderr)
        print("No files from this run will be marked as processed due to the error.", file=sys.stderr)
        # Note: successfully_loaded_files are not logged if this overall try block fails.

    print("-----------------------------------------")
    print("Incremental ingestion complete!")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
