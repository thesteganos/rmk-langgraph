import os
import json
import sys # Added for stderr
from datetime import datetime
# Removed HuggingFaceEmbeddings import, will get from src.utils
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.utils import get_embedding_model # Added import for centralized model loading

# Note: This tool can optionally use the LLM to process knowledge gaps.
# We import the necessary components for that advanced functionality.
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import pubmed_tool

# --- Configuration ---
load_dotenv()
DB_PATH = "db"
PENDING_REVIEW_FILE = "pending_review.jsonl"
USER_FEEDBACK_FILE = "feedback_for_review.jsonl"
KNOWLEDGE_GAPS_FILE = "knowledge_gaps.jsonl"
PROCESSED_DIR = "processed_logs"

def get_llm_with_tools():
    """Initializes the LLM and binds it with tools for processing knowledge gaps."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not google_api_key or not model_name:
        print("WARNING: Cannot process knowledge gaps without GOOGLE_API_KEY and LLM_MODEL in .env")
        return None
    
    llm_instance = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=google_api_key)
    tools = [llm_instance.get_tools("google_search_retrieval")[0], pubmed_tool]
    return llm_instance.bind_tools(tools)

def process_propositions(db):
    """Processes new knowledge proposed by the automated knowledge_pipeline.py."""
    print("\n--- Processing Automated Knowledge Propositions ---")
    propositions = []
    try:
        with open(PENDING_REVIEW_FILE, "r") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    propositions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from {PENDING_REVIEW_FILE} at line {line_number}: {e}", file=sys.stderr)
                    # Optionally, decide if you want to skip the line or stop processing the file
    except FileNotFoundError:
        print(f"Info: No pending propositions file found at {PENDING_REVIEW_FILE}. Skipping.")
        return 0
    except IOError as e:
        print(f"Error: Could not read file {PENDING_REVIEW_FILE}: {e}", file=sys.stderr)
        return 0


    if not propositions:
        print("No new knowledge propositions to review.")
        return 0

    approved_texts = []
    for i, prop in enumerate(propositions):
        print(f"\n{'='*80}\nPROPOSITION {i+1}/{len(propositions)}\nSOURCE: {prop['source_url']}\n\nAI-Generated Question:\n{prop['question']}\n\nAI-Generated Answer:\n{prop['answer']}\n{'='*80}")
        action = input("Action: [a]pprove, [s]kip, [q]uit? ").lower()
        if action == 'q': break
        if action == 'a':
            text_to_add = f"Source: {prop['source_url']}\n\nQuestion: {prop['question']}\n\nVerified Answer: {prop['answer']}"
            approved_texts.append(text_to_add)
            print("--> Approved.")
        else: print("--> Skipped.")
    
    if approved_texts:
        db.add_texts(texts=approved_texts)
    
    # Archive the processed file
    try:
        if os.path.exists(PENDING_REVIEW_FILE): # Check if file exists before trying to rename
            archive_name = os.path.join(PROCESSED_DIR, f"propositions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            os.rename(PENDING_REVIEW_FILE, archive_name)
            print(f"Archived {PENDING_REVIEW_FILE} to {archive_name}")
    except OSError as e:
        print(f"Error: Could not archive file {PENDING_REVIEW_FILE}: {e}", file=sys.stderr)
    return len(approved_texts)

def process_user_feedback(db):
    """Processes 'Good' Q&A pairs submitted by users via the web app."""
    print("\n--- Processing 'Good' User Feedback ---")
    feedback_entries = []
    try:
        with open(USER_FEEDBACK_FILE, "r") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    feedback_entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from {USER_FEEDBACK_FILE} at line {line_number}: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Info: No user feedback file found at {USER_FEEDBACK_FILE}. Skipping.")
        return 0
    except IOError as e:
        print(f"Error: Could not read file {USER_FEEDBACK_FILE}: {e}", file=sys.stderr)
        return 0

    good_feedback = [e for e in feedback_entries if e["feedback"] == "good"]
    if not good_feedback:
        print("No new 'good' user feedback to process.")
        return 0

    approved_texts = []
    for i, entry in enumerate(good_feedback):
        print(f"\n{'='*80}\nUSER FEEDBACK {i+1}/{len(good_feedback)}\nQUERY: {entry['interaction']['query']}\n\nANSWER:\n{entry['interaction']['answer']}\n{'='*80}")
        action = input("Action: [a]pprove this Q&A, [s]kip, [q]uit? ").lower()
        if action == 'q': break
        if action == 'a':
            text_to_add = f"User Query: {entry['interaction']['query']}\n\nVerified Answer: {entry['interaction']['answer']}"
            approved_texts.append(text_to_add)
            print("--> Approved.")
        else: print("--> Skipped.")

    if approved_texts:
        db.add_texts(texts=approved_texts)

    # Archive the processed file
    try:
        if os.path.exists(USER_FEEDBACK_FILE):
            archive_name = os.path.join(PROCESSED_DIR, f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            os.rename(USER_FEEDBACK_FILE, archive_name)
            print(f"Archived {USER_FEEDBACK_FILE} to {archive_name}")
    except OSError as e:
        print(f"Error: Could not archive file {USER_FEEDBACK_FILE}: {e}", file=sys.stderr)
    return len(approved_texts)

def process_knowledge_gaps(db, llm_with_tools):
    """Processes queries where the trusted RAG failed, then asks for verification."""
    print("\n--- Processing Logged Knowledge Gaps ---")
    if not llm_with_tools:
        print("Skipping knowledge gap processing because LLM is not configured.")
        return 0
        
    gap_entries = []
    try:
        with open(KNOWLEDGE_GAPS_FILE, "r") as f:
            for line_number, line in enumerate(f, 1):
                try:
                    gap_entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from {KNOWLEDGE_GAPS_FILE} at line {line_number}: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Info: No knowledge gap file found at {KNOWLEDGE_GAPS_FILE}. Skipping.")
        return 0
    except IOError as e:
        print(f"Error: Could not read file {KNOWLEDGE_GAPS_FILE}: {e}", file=sys.stderr)
        return 0

    if not gap_entries:
        print("No new knowledge gaps to process.")
        return 0

    approved_texts = []
    for i, entry in enumerate(gap_entries):
        query = entry['query']
        print(f"\n{'='*80}\nKNOWLEDGE GAP {i+1}/{len(gap_entries)}\nQUERY: {query}\n{'='*80}")
        print("Searching for an answer using web tools...")
        
        # Use the agent to generate a proposed answer
        proposed_answer = llm_with_tools.invoke(query).content
        
        print(f"\nPROPOSED ANSWER:\n{proposed_answer}")
        action = input("\nAction: [a]pprove this new Q&A, [s]kip, [q]uit? ").lower()
        
        if action == 'q': break
        if action == 'a':
            text_to_add = f"User Query: {query}\n\nVerified Answer: {proposed_answer}"
            approved_texts.append(text_to_add)
            print("--> Approved.")
        else: print("--> Skipped.")

    if approved_texts:
        db.add_texts(texts=approved_texts)

    # Archive the processed file
    try:
        if os.path.exists(KNOWLEDGE_GAPS_FILE):
            archive_name = os.path.join(PROCESSED_DIR, f"gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
            os.rename(KNOWLEDGE_GAPS_FILE, archive_name)
            print(f"Archived {KNOWLEDGE_GAPS_FILE} to {archive_name}")
    except OSError as e:
        print(f"Error: Could not archive file {KNOWLEDGE_GAPS_FILE}: {e}", file=sys.stderr)
    return len(approved_texts)

def main():
    """Main function to run the expert review and ingestion workflow."""
    print("--- Expert Knowledge Verification & Ingestion Tool ---")
    
    # Create archive directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Initialize components
    print("Loading embedding model...")
    embeddings = get_embedding_model()
    if not embeddings:
        print("FATAL: Embedding model could not be loaded. Exiting.", file=sys.stderr)
        return # Exit main()
    print("Embedding model loaded successfully.")

    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm_with_tools = get_llm_with_tools()
    print(f"Connected to ChromaDB at {DB_PATH}")

    # Process all three sources of new knowledge
    prop_added = process_propositions(db)
    feedback_added = process_user_feedback(db)
    gaps_added = process_knowledge_gaps(db, llm_with_tools)
    
    total_added = prop_added + feedback_added + gaps_added
    if total_added > 0:
        print(f"\n--- WORKFLOW COMPLETE ---")
        print(f"Knowledge base successfully updated with {total_added} new document(s)!")
    else:
        print("\n--- WORKFLOW COMPLETE ---")
        print("No new knowledge was added to the database.")

if __name__ == "__main__":
    main()
