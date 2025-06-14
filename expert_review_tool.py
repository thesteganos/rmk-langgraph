import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

DB_PATH = "db"
PENDING_REVIEW_FILE = "pending_review.jsonl"
PROCESSED_FILE = "processed_feedback.jsonl"

def main():
    print("--- Expert Knowledge Verification Tool ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print(f"Connected to ChromaDB at {DB_PATH}")

    try:
        with open(PENDING_REVIEW_FILE, "r") as f:
            propositions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("No pending review file found. Exiting.")
        return

    if not propositions:
        print("No new knowledge propositions to review.")
        return

    approved_texts_for_ingestion = []
    
    for i, prop in enumerate(propositions):
        print(f"\n{'='*80}\nPROPOSITION {i+1}/{len(propositions)}\nSOURCE: {prop['source_url']}\n\nAI-Generated Question:\n{prop['question']}\n\nAI-Generated Answer:\n{prop['answer']}\n{'='*80}")
        action = input("Action: [a]pprove, [s]kip, [q]uit? ").lower()
        if action == 'q': break
        if action == 'a':
            text_to_add = f"Source: {prop['source_url']}\n\nQuestion: {prop['question']}\n\nVerified Answer: {prop['answer']}"
            approved_texts_for_ingestion.append(text_to_add)
            print("--> Approved. Staged for ingestion.")
        else: print("--> Skipped.")

    if approved_texts_for_ingestion:
        print(f"\nAdding {len(approved_texts_for_ingestion)} new verified document(s) to ChromaDB...")
        db.add_texts(texts=approved_texts_for_ingestion)
        print("Knowledge base successfully updated!")
    
    with open(PROCESSED_FILE, "a") as f_proc:
        for prop in propositions: f_proc.write(json.dumps(prop) + "\n")
    os.remove(PENDING_REVIEW_FILE)
    print(f"\nProcessed propositions moved to {PROCESSED_FILE}.")

if __name__ == "__main__":
    main()
