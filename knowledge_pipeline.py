import os
import requests
from bs4 import BeautifulSoup
import json
import sys # Added for stderr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TRUSTED_SOURCES = {
    "WHO_Nutrition": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "MayoClinic_Metabolism": "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/metabolism/art-20046508",
    "NIH_Dietary_Supplements": "https://ods.od.nih.gov/factsheets/WeightLoss-HealthProfessional/"
}
PENDING_REVIEW_FILE = "pending_review.jsonl"
PROCESSED_URLS_LOG = "processed_urls.log"

def get_processed_urls():
    """Reads the log of already processed URLs."""
    try:
        with open(PROCESSED_URLS_LOG, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        # If the log file doesn't exist, it means no URLs have been processed yet.
        return set()
    except IOError as e:
        print(f"Error: Could not read processed URLs log {PROCESSED_URLS_LOG}: {e}", file=sys.stderr)
        # Depending on desired behavior, could exit or return empty set and try to continue
        return set()


def log_processed_url(url):
    """Adds a new URL to the processed log."""
    try:
        with open(PROCESSED_URLS_LOG, "a") as f:
            f.write(url + "\n")
    except IOError as e:
        print(f"Error: Could not write to processed URLs log {PROCESSED_URLS_LOG}: {e}", file=sys.stderr)


def get_article_text(url: str) -> str:
    # ... (get_article_text function remains the same as previous version) ...
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('article') or soup.find(id='main-content') or soup.find(role='main') or soup.body
        if main_content:
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            return ' '.join(p.get_text(strip=True) for p in main_content.find_all('p'))
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    print("--- Starting Automated Knowledge Pipeline ---")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not google_api_key or not model_name:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY and LLM_MODEL must be set in .env")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a medical knowledge extraction system. Read the article text and generate a single, clear question it answers, and a detailed answer based ONLY on the text.
        Your output MUST be a single, valid JSON object with keys "question" and "answer". Do not include any other text or formatting.
        ARTICLE TEXT: --- {article_text} ---"""
    )
    extraction_chain = prompt | llm | StrOutputParser()

    processed_urls = get_processed_urls()
    new_articles_to_process = []

    print("Checking sources for new articles...")
    for source_name, url in TRUSTED_SOURCES.items():
        # In a real system, you'd scrape a listing page. Here we check the direct URL.
        if url not in processed_urls:
            print(f"Found new URL to process: {url}")
            article_text = get_article_text(url)
            if article_text:
                new_articles_to_process.append({"text": article_text, "url": url})
    
    if not new_articles_to_process:
        print("No new articles found. Pipeline is up-to-date.")
        return

    print(f"Found {len(new_articles_to_process)} new articles. Processing in parallel...")
    
    # --- PERFORMANCE: Use .batch() for parallel processing ---
    batch_inputs = [{"article_text": article["text"]} for article in new_articles_to_process]
    batch_results = extraction_chain.batch(batch_inputs)

    for i, result_str in enumerate(batch_results):
        url = new_articles_to_process[i]["url"]
        try:
            proposition = json.loads(result_str) # This can raise JSONDecodeError
            proposition['source_url'] = url

            try:
                with open(PENDING_REVIEW_FILE, "a") as f:
                    f.write(json.dumps(proposition) + "\n")
                log_processed_url(url) # Log as processed only after successful write and LLM parse
                print(f"--> Proposition saved for review from {url}")
            except IOError as e:
                print(f"Error: Could not write to pending review file {PENDING_REVIEW_FILE} for URL {url}: {e}", file=sys.stderr)
                # Decide if we should attempt to remove url from processed_urls if logging it happened before this error.
                # Current logic logs after write, so that's fine.

        except json.JSONDecodeError as e:
            print(f"Error: FAILED to parse LLM response for {url}: {e}. LLM Output: '{result_str}'. Skipping.", file=sys.stderr)
    
    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()
