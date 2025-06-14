import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
DB_PATH = "db"

def main():
    """Creates a ChromaDB vector store from PDF documents in the DATA_PATH."""
    print("Starting ingestion process...")

    # --- FIX: Check if the data directory is empty before proceeding ---
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"The '{DATA_PATH}' directory is empty or does not exist.")
        print("Please add your trusted PDF files to it before running ingestion.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # --- FIX: Force model to run on CPU for wider compatibility ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    print("Embeddings model loaded (running on CPU).")

    print("Creating and persisting vector store...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_PATH
    )
    print("-----------------------------------------")
    print(f"Ingestion complete! Vector store created at {DB_PATH}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
