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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    prompt = ChatPromptTemplate.from_template(
        """Read the following article. Generate a single, clear question the article answers, and a detailed answer based ONLY on the text.
        Format your response as a JSON object with keys "question" and "answer".
        ARTICLE TEXT: --- {article_text} ---"""
    )
    extraction_chain = prompt | llm | StrOutputParser()

    for source_name, url in TRUSTED_SOURCES.items():
        print(f"\nProcessing source: {source_name}")
        article_text = get_article_text(url)
        if not article_text: continue
            
        response_str = extraction_chain.invoke({"article_text": article_text})
        try:
            json_response_str = response_str.strip().replace("```json", "").replace("```", "")
            proposition = json.loads(json_response_str)
            proposition['source_url'] = url
            with open(PENDING_REVIEW_FILE, "a") as f:
                f.write(json.dumps(proposition) + "\n")
            print(f"--> Proposition saved for review: Q: {proposition['question'][:50]}...")
        except json.JSONDecodeError as e:
            print(f"--> Failed to parse LLM response as JSON: {e}")

    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()
