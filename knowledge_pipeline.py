import os
import requests
from bs4 import BeautifulSoup
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

TRUSTED_SOURCES = {
    "WHO_Nutrition": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "MayoClinic_Metabolism": "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/metabolism/art-20046508"
}
PENDING_REVIEW_FILE = "pending_review.jsonl"

def get_article_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find(id='main-content') or soup.find('article') or soup.body
        return ' '.join(p.get_text() for p in main_content.find_all('p'))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    print("--- Starting Automated Knowledge Pipeline ---")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file. Pipeline cannot start.")

    # --- NEW: Use the single, configurable model ---
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise ValueError("FATAL ERROR: LLM_MODEL not defined in .env file. Pipeline cannot start.")
        
    print(f"INFO: Using LLM_MODEL for pipeline: {model_name}")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
    # --- END NEW CODE ---
    
    prompt = ChatPromptTemplate.from_template(
        """You are a medical knowledge extraction system. Read the following article text and perform two tasks:
        1.  Generate a single, clear question that the article answers.
        2.  Generate a detailed, comprehensive answer to that question based ONLY on the provided text.
        Your final output MUST be a single, valid JSON object with two keys: "question" and "answer". Do not include any other text or formatting like "```json".
        Example:
        {{"question": "What are the benefits of a healthy diet?", "answer": "A healthy diet helps protect against malnutrition..."}}
        ARTICLE TEXT:
        ---
        {article_text}
        ---
        """
    )
    
    extraction_chain = prompt | llm | StrOutputParser()

    for source_name, url in TRUSTED_SOURCES.items():
        # ... (rest of the script remains the same) ...
        print(f"\nProcessing source: {source_name}")
        article_text = get_article_text(url)
        if not article_text: continue
            
        response_str = extraction_chain.invoke({"article_text": article_text})
        try:
            proposition = json.loads(response_str)
            proposition['source_url'] = url
            with open(PENDING_REVIEW_FILE, "a") as f:
                f.write(json.dumps(proposition) + "\n")
            print(f"--> Proposition saved for review: Q: {proposition['question'][:50]}...")
        except json.JSONDecodeError as e:
            print(f"--> Failed to parse LLM response as JSON: {e}")

    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()
