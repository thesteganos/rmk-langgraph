"""
Utility functions for the knowledge management system.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

def get_embedding_model():
    """
    Initializes and returns the HuggingFace sentence transformer embedding model.
    Model: "all-MiniLM-L6-v2"
    Device: CPU
    """
    try:
        # Initialize the HuggingFace embeddings model
        # Using a sentence transformer model that runs well on CPU
        model_name = "all-MiniLM-L6-v2"
        # Specify that the model should run on CPU
        model_kwargs = {'device': 'cpu'}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        return embeddings
    except Exception as e:
        print(f"Error: Failed to load HuggingFace embedding model '{model_name}': {e}", file=sys.stderr)
        # Depending on the desired behavior, you might re-raise the exception
        # or exit the program if embeddings are critical.
        # For now, we'll let the caller handle a None return or re-raise.
        raise  # Re-raise the exception to make it clear that loading failed

if __name__ == '__main__':
    # Example usage:
    try:
        print("Attempting to load embedding model...")
        model = get_embedding_model()
        if model:
            print("Embedding model loaded successfully.")
            # You could try a sample embedding here if desired
            # sample_text = "This is a test sentence."
            # sample_embedding = model.embed_query(sample_text)
            # print(f"Sample embedding for '{sample_text}': {sample_embedding[:5]}...") # Print first 5 dimensions
        else:
            print("Failed to load embedding model (model is None).")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
