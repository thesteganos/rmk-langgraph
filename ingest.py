import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
DB_PATH = "db"
PROCESSED_LOG_FILE = os.path.join(DB_PATH, "processed_files.log")

def get_processed_files():
    """Reads the log of already processed files."""
    if not os.path.exists(PROCESSED_LOG_FILE):
        return set()
    with open(PROCESSED_LOG_FILE, "r") as f:
        return set(line.strip() for line in f)

def log_processed_file(filename):
    """Adds a new file to the processed log."""
    with open(PROCESSED_LOG_FILE, "a") as f:
        f.write(filename + "\n")

def main():
    """Performs incremental ingestion of new PDF files into the ChromaDB vector store."""
    print("--- Starting Incremental Ingestion Process ---")
    os.makedirs(DB_PATH, exist_ok=True)

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"The '{DATA_PATH}' directory is empty. Please add PDF files to ingest.")
        return

    processed_files = get_processed_files()
    all_files = set(os.listdir(DATA_PATH))
    new_files = sorted(list(all_files - processed_files))

    if not new_files:
        print("No new documents to process. Knowledge base is up-to-date.")
        return

    print(f"Found {len(new_files)} new document(s) to process: {new_files}")

    # Process only the new files
    # To do this, we temporarily move other files, process, then move them back.
    # A more robust way would be to pass a list of files to the loader if it supports it.
    # This workaround is effective for PyPDFDirectoryLoader.
    
    temp_dir = "temp_ingest"
    os.makedirs(temp_dir, exist_ok=True)

    # Move processed files temporarily
    for filename in processed_files:
        if os.path.exists(os.path.join(DATA_PATH, filename)):
            os.rename(os.path.join(DATA_PATH, filename), os.path.join(temp_dir, filename))

    try:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        if not documents:
            print("Loader returned no documents. Check PDF files for issues.")
            return

        print(f"Loaded {len(documents)} new document(s).")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        print("Embeddings model loaded (running on CPU).")

        print("Adding new documents to the vector store...")
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        db.add_documents(texts)
        print("Knowledge base updated successfully.")

        # Log the newly processed files
        for filename in new_files:
            log_processed_file(filename)
        print("Updated the processed files log.")

    finally:
        # Move processed files back
        for filename in os.listdir(temp_dir):
            os.rename(os.path.join(temp_dir, filename), os.path.join(DATA_PATH, filename))
        os.rmdir(temp_dir)

    print("-----------------------------------------")
    print("Incremental ingestion complete!")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
