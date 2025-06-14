import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

DB_PATH = "db"
PENDING_REVIEW_FILE = "pending_review.jsonl"
USER_FEEDBACK_FILE = "feedback_for_review.jsonl"
PROCESSED_DIR = "processed_logs"

def process_propositions(db):
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
    
    # Move processed file
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.rename(PENDING_REVIEW_FILE, os.path.join(PROCESSED_DIR, f"propositions_{len(propositions)}.jsonl"))
    return len(approved_texts)

def process_user_feedback(db):
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

    # Move processed file
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.rename(USER_FEEDBACK_FILE, os.path.join(PROCESSED_DIR, f"user_feedback_{len(feedback_entries)}.jsonl"))
    return len(approved_texts)

def process_knowledge_gaps(db):
    print("\n--- Processing Logged Knowledge Gaps ---")
    try:
        with open("knowledge_gaps.jsonl", "r") as f:
            gap_entries = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("No knowledge gap file found. Skipping.")
        return 0

    if not gap_entries:
        print("No new knowledge gaps to process.")
        return 0

    print("Found gaps. These will be run through the knowledge pipeline to find answers.")
    # Here, we would integrate with the logic from knowledge_pipeline.py
    # For simplicity, we will just print them. A full implementation would
    # call a function that takes a query, scrapes the web, and generates a proposition.
    for entry in gap_entries:
        print(f"  - GAP DETECTED FOR QUERY: {entry['query']}")
        # TODO: Trigger knowledge_pipeline logic for this query
    
    # After processing, move the file
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.rename("knowledge_gaps.jsonl", os.path.join(PROCESSED_DIR, f"gaps_{len(gap_entries)}.jsonl"))
    print(f"Knowledge gaps have been logged for automated processing.")
    return 0 # We are not adding directly, just logging for now.

def main():
    print("--- Expert Knowledge Verification & Ingestion Tool ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print(f"Connected to ChromaDB at {DB_PATH}")

    # --- FIX: Process both feedback sources ---
    prop_added = process_propositions(db)
    feedback_added = process_user_feedback(db)
    gaps_processed = process_knowledge_gaps(db)
    
    total_added = prop_added + feedback_added
    if total_added > 0:
        print(f"\nKnowledge base successfully updated with {total_added} new document(s)!")
    else:
        print("\nNo new knowledge was added to the database.")

if __name__ == "__main__":
    main()
