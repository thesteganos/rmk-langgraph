import os
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv

# LangChain components for interacting with the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
# This is the "watchlist" of high-quality websites the pipeline will monitor.
# For a production system, this list would be much larger and more specific.
TRUSTED_SOURCES = {
    "WHO_Nutrition": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "MayoClinic_Metabolism": "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/metabolism/art-20046508",
    "NIH_Dietary_Supplements": "https://ods.od.nih.gov/factsheets/WeightLoss-HealthProfessional/"
}

# The output file where proposed new knowledge will be saved for review.
PENDING_REVIEW_FILE = "pending_review.jsonl"

# --- Helper Function ---

def get_article_text(url: str) -> str:
    """
    Fetches and extracts the main text content from a given URL.
    Includes basic error handling for network issues or missing content.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # A simple but effective way to find the main content area in many articles
        main_content = soup.find('article') or soup.find(id='main-content') or soup.find(role='main') or soup.body
        
        if main_content:
            # Remove script and style elements to clean up the text
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            return ' '.join(p.get_text(strip=True) for p in main_content.find_all('p'))
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# --- Main Execution ---

def main():
    """
    Main function to run the automated knowledge pipeline.
    It scrapes trusted sources, uses an LLM to generate Q&A pairs,
    and saves them as propositions for expert review.
    """
    print("--- Starting Automated Knowledge Pipeline ---")

    # --- Robust Startup Checks ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not google_api_key:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file. Pipeline cannot start.")
    if not model_name:
        raise ValueError("FATAL ERROR: LLM_MODEL not defined in .env file. Pipeline cannot start.")
        
    print(f"INFO: Using LLM_MODEL for pipeline: {model_name}")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
    
    # --- Robust Prompting ---
    # This prompt is engineered to force the LLM to return clean, valid JSON.
    prompt = ChatPromptTemplate.from_template(
        """You are a medical knowledge extraction system. Read the following article text and perform two tasks:
        1.  Generate a single, clear question that the article answers.
        2.  Generate a detailed, comprehensive answer to that question based ONLY on the provided text.

        Your final output MUST be a single, valid JSON object with two keys: "question" and "answer". Do not include any other text, comments, or formatting like "```json".

        Example of a valid response:
        {{"question": "What are the benefits of a healthy diet?", "answer": "A healthy diet helps protect against malnutrition in all its forms..."}}

        ARTICLE TEXT:
        ---
        {article_text}
        ---
        """
    )
    
    extraction_chain = prompt | llm | StrOutputParser()

    # --- Processing Loop ---
    for source_name, url in TRUSTED_SOURCES.items():
        print(f"\nProcessing source: {source_name} ({url})")
        article_text = get_article_text(url)
        
        if not article_text:
            print("--> Could not retrieve article text. Skipping.")
            continue
            
        response_str = extraction_chain.invoke({"article_text": article_text})
        
        try:
            # Attempt to parse the LLM's response string into a Python dictionary
            proposition = json.loads(response_str)
            
            # Add the source URL for verification in the review tool
            proposition['source_url'] = url
            
            # Append the valid proposition to the review file
            with open(PENDING_REVIEW_FILE, "a") as f:
                f.write(json.dumps(proposition) + "\n")
            print(f"--> Proposition saved for review: Q: {proposition.get('question', 'N/A')[:60]}...")
            
        except json.JSONDecodeError:
            print(f"--> FAILED to parse LLM response as JSON. Skipping this entry.")
            print(f"--> Raw response for debugging: {response_str}")
        except Exception as e:
            print(f"--> An unexpected error occurred: {e}")

    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()
