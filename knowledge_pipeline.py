import os
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import sys
from dotenv import load_dotenv
from urllib.parse import urljoin, quote_plus, urlparse, urlunparse

# LangChain components for interacting with the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---

# 1. EXPANDED TRUSTED SOURCES (for passive, homepage-style scraping)
# We keep this for monitoring specific, high-value pages.
TRUSTED_SOURCES = {
    "WHO_Healthy_Diet": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "NIH_Supplements_Factsheet": "https://ods.od.nih.gov/factsheets/WeightLoss-HealthProfessional/"
}

# 2. TERMS OF INTEREST (for active, recursive knowledge seeking)
# The pipeline will proactively search for these topics.
TERMS_OF_INTEREST = [
    "semaglutide for weight loss",
    "metformin and weight management",
    "creatine monohydrate benefits",
    "sarcopenia diagnosis and treatment",
    "intermittent fasting science",
    "ketogenic diet for obesity",
    "protein intake for muscle synthesis",
    "ozempic side effects",
    "GLP-1 Receptor Agonists",
    "Dual GIP and GLP-1 Receptor Agonists",
    "Phentermine/Topiramate",
    "Naltrexone/Bupropion",
    "Orlistat",
    "Bariatric Surgery",
    "Gastric Bypass (Roux-en-Y)",
    "Sleeve Gastrectomy",
    "Gastric Banding",
    "Medical Nutrition Therapy (MNT)",
    "Behavioral Therapy",
    "Comprehensive Lifestyle Intervention",
    "Calorie Deficit",
    "Macronutrient Balance",
    "Low-Carbohydrate Diets",
    "Intermittent Fasting (IF)",
    "Mediterranean Diet",
    "Volumetrics",
    "Energy Balance",
    "Total Daily Energy Expenditure (TDEE)",
    "Basal Metabolic Rate (BMR)",
    "Weight Loss Plateau",
    "Body Composition",
    "Resistance Training",
    "Progressive Overload",
    "Mechanical Tension",
    "Muscle Damage",
    "Metabolic Stress",
    "Protein Synthesis",
    "Caloric Surplus",
    "Leucine",
    "Nutrient Timing",
    "Creatine Monohydrate",
    "Metabolic Syndrome (MetS)",
    "Insulin Resistance",
    "Type 2 Diabetes",
    "Dyslipidemia",
    "Non-alcoholic Fatty Liver Disease (NAFLD)",
    "Glycemic Control",
    "HbA1c",
    "Lipid Panel",
    "Cardiometabolic Health",
    "Sarcopenic Obesity",
]

# 3. TRUSTED SEARCH SITES (the search engines of our trusted sources)
# The pipeline will use these templates to construct search queries.
TRUSTED_SEARCH_SITES = {
    "Mayo Clinic Search": "https://www.mayoclinic.org/search/search-results?q={query}",
    "Examine.com Search": "https://examine.com/search/?q={query}",
    "NIH Search": "https://www.nih.gov/search/results?keys={query}",
    "Cleveland Clinic Search": "https://my.clevelandclinic.org/health/search?q={query}",
    "Johns Hopkins Search": "https://www.hopkinsmedicine.org/search?query={query}"
}

# --- File Paths ---
PENDING_REVIEW_FILE = "pending_review.jsonl"
PROCESSED_URLS_LOG = "processed_urls.log"
LLM_PARSING_ERRORS_LOG = "llm_parsing_errors.log"

# --- Helper Functions ---

def get_processed_urls():
    """Reads the log of URLs that have already been processed to avoid duplicates."""
    if not os.path.exists(PROCESSED_URLS_LOG):
        return set()
    try:
        with open(PROCESSED_URLS_LOG, "r") as f:
            return set(line.strip() for line in f)
    except IOError as e:
        print(f"CRITICAL Error: Could not read processed URLs log {PROCESSED_URLS_LOG}: {e}", file=sys.stderr)
        raise  # Re-raise the exception to halt execution if this critical file can't be read

def log_processed_url(url):
    """Adds a new URL to the processed log."""
    try:
        with open(PROCESSED_URLS_LOG, "a") as f:
            f.write(url + "\n")
    except IOError as e:
        print(f"CRITICAL Error: Could not write to processed URLs log {PROCESSED_URLS_LOG} for URL {url}: {e}", file=sys.stderr)
        raise # Re-raise to indicate failure

def get_article_text(url: str) -> str:
    """Fetches and extracts the main text content from a given article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
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
        print(f"Error fetching article {url}: {e}")
        return None

def scrape_search_results(search_url: str, base_url: str) -> list[str]:
    """Scrapes a search results page and extracts the top article links."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        # This is a generic selector; it might need tuning for specific sites.
        # It looks for links within heading tags, a common pattern.
        for tag in soup.select('h2 a, h3 a, h4 a'):
            if 'href' in tag.attrs:
                # Convert relative URLs (e.g., /page.html) to absolute URLs
                full_url = urljoin(base_url, tag['href'])
                if full_url not in links:
                    links.append(full_url)
            if len(links) >= 3: # Limit to top 3 results per search
                break
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error scraping search results {search_url}: {e}")
        return []

from urllib.parse import urlparse, urlunparse, quote_plus # Ensure quote_plus is here if it was only in main

def generate_full_search_url_and_base(search_template: str, query: str) -> tuple[str, str]:
    """
    Generates the full search URL and the base URL from a template and query.
    Example:
    search_template = "https://www.example.com/search?q={query}"
    query = "test query"
    Returns: ("https://www.example.com/search?q=test+query", "https://www.example.com")
    """
    encoded_query = quote_plus(query)
    full_search_url = search_template.format(query=encoded_query)

    parsed_search_url = urlparse(full_search_url)
    base_url = urlunparse((parsed_search_url.scheme, parsed_search_url.netloc, '', '', '', ''))
    return full_search_url, base_url

# --- Main Execution ---

def main():
    """Main function to run the upgraded, proactive knowledge pipeline."""
    print("--- Starting Automated & Recursive Knowledge Pipeline ---")
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
    urls_already_targeted_this_run = set() # New set

    # --- Phase 1: Passive Scraping of High-Value Static Pages ---
    print("\n--- Phase 1: Checking Static Trusted Sources ---")
    for source_name, url in TRUSTED_SOURCES.items():
        if url not in processed_urls and url not in urls_already_targeted_this_run: # Modified condition
            print(f"Found new static URL to process: {url}")
            article_text = get_article_text(url)
            if article_text:
                new_articles_to_process.append({"text": article_text, "url": url})
                urls_already_targeted_this_run.add(url) # Add to the new set

    # --- Phase 2: Active & Recursive Searching for Terms of Interest ---
    print("\n--- Phase 2: Actively Searching for Terms of Interest ---")
    for term in TERMS_OF_INTEREST:
        print(f"\n-- Searching for term: '{term}' --")
        for site_name, search_template in TRUSTED_SEARCH_SITES.items():
            search_url, base_url = generate_full_search_url_and_base(search_template, term)

            print(f"  > Searching on {site_name}...")
            article_urls = scrape_search_results(search_url, base_url)

            for url in article_urls:
                if url not in processed_urls and url not in urls_already_targeted_this_run: # Modified condition
                    print(f"    * Found new article link: {url}")
                    article_text = get_article_text(url)
                    if article_text:
                        new_articles_to_process.append({"text": article_text, "url": url})
                        urls_already_targeted_this_run.add(url) # Add to the new set

    # --- Final Batch Processing (unified for both phases) ---
    if not new_articles_to_process:
        print("\nNo new articles found. Pipeline is up-to-date.")
        return

    print(f"\n--- Found {len(new_articles_to_process)} total new articles. Processing in parallel... ---")
    
    batch_inputs = [{"article_text": article["text"]} for article in new_articles_to_process]
    # Limit concurrency to be kind to the API
    # Load MAX_CONCURRENCY from environment variable, default to 5
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "5"))
    batch_results = extraction_chain.batch(batch_inputs, {"max_concurrency": max_concurrency})

    for i, result_str in enumerate(batch_results):
        url = new_articles_to_process[i]["url"]
        try:
            proposition = json.loads(result_str)
            proposition['source_url'] = url
            with open(PENDING_REVIEW_FILE, "a") as f:
                f.write(json.dumps(proposition) + "\n")
            log_processed_url(url) # Log as processed only after successful proposition
            print(f"--> Proposition saved for review from {url}")
        except json.JSONDecodeError as e: # Added 'as e'
            error_message = f"--> FAILED to parse LLM response for {url}. Raw output (first 500 chars): '{result_str[:500]}...' Error: {e}. Skipping."
            print(error_message, file=sys.stderr) # Print to stderr for consistency

            # Log the full error details to a separate file
            try:
                error_log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "url": url,
                    "error_type": "JSONDecodeError",
                    "message": str(e),
                    "full_llm_output": result_str
                }
                # LLM_PARSING_ERRORS_LOG needs to be in scope in the modified file's main()
                with open(LLM_PARSING_ERRORS_LOG, "a") as error_f:
                    error_f.write(json.dumps(error_log_entry) + "\n")
            except Exception as log_e:
                print(f"CRITICAL: Failed to write to {LLM_PARSING_ERRORS_LOG}: {log_e}", file=sys.stderr)
    
    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()
