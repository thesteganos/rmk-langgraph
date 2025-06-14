import os
import json
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

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
    try:
        with open(PENDING_REVIEW_FILE, "r") as f:
            propositions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("No pending propositions file found. Skipping.")
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
    os.rename(PENDING_REVIEW_FILE, os.path.join(PROCESSED_DIR, f"propositions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"))
    return len(approved_texts)

def process_user_feedback(db):
    """Processes 'Good' Q&A pairs submitted by users via the web app."""
    print("\n--- Processing 'Good' User Feedback ---")
    try:
        with open(USER_FEEDBACK_FILE, "r") as f:
            feedback_entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("No user feedback file found. Skipping.")
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
    os.rename(USER_FEEDBACK_FILE, os.path.join(PROCESSED_DIR, f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"))
    return len(approved_texts)

def process_knowledge_gaps(db, llm_with_tools):
    """Processes queries where the trusted RAG failed, then asks for verification."""
    print("\n--- Processing Logged Knowledge Gaps ---")
    if not llm_with_tools:
        print("Skipping knowledge gap processing because LLM is not configured.")
        return 0
        
    try:
        with open(KNOWLEDGE_GAPS_FILE, "r") as f:
            gap_entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("No knowledge gap file found. Skipping.")
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
    os.rename(KNOWLEDGE_GAPS_FILE, os.path.join(PROCESSED_DIR, f"gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"))
    return len(approved_texts)

def main():
    """Main function to run the expert review and ingestion workflow."""
    print("--- Expert Knowledge Verification & Ingestion Tool ---")
    
    # Create archive directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Initialize components
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
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
