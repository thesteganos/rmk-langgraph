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
    # Core Focus: Obesity, Weight Loss & Metabolic Disorders
    # Obesity & Weight Management Strategies
    "Obesity diagnosis and assessment",
    "Obesity comorbidities",
    "Pharmacotherapy for obesity guidelines", # New
    "Semaglutide for weight loss and diabetes (Ozempic, Wegovy, Rybelsus)",
    "Liraglutide (Saxenda, Victoza)", # Existing, re-prioritized
    "liraglutide for weight loss", # New & specific
    "Tirzepatide (Mounjaro, Zepbound)", # Existing, re-prioritized
    "tirzepatide for weight loss", # New & specific
    "Setmelanotide for genetic obesity", # New
    "Retatrutide clinical trials",
    "GLP-1 Receptor Agonists mechanisms and side effects",
    "Metformin for weight management and prediabetes",
    "Phentermine/Topiramate (Qsymia) for obesity",
    "Naltrexone/Bupropion (Contrave) for obesity",
    "Orlistat (Alli, Xenical) for weight loss",
    "Anti-obesity medications pipeline and future research",
    "Bariatric Surgery criteria, types, and outcomes",
    "Gastric Bypass (Roux-en-Y) surgery",
    "Sleeve Gastrectomy surgery",
    "Gastric Banding adjustments and complications",
    "Endoscopic sleeve gastroplasty procedure",
    "Calorie Deficit for weight loss",
    "Energy Balance equation",
    "Total Daily Energy Expenditure (TDEE) calculation",
    "Basal Metabolic Rate (BMR) factors",
    "Weight Loss Plateau causes and solutions",
    "Body Composition Analysis methods (DEXA, BIA)",
    "Visceral fat reduction", # New
    "Set Point Theory of weight regulation",
    "Appetite regulation and obesity", # New
    "Gut microbiome and obesity", # New
    "Medical Nutrition Therapy (MNT) for obesity",
    "Ketogenic diet for obesity and metabolic health",
    "Low-Carbohydrate Diets efficacy and safety",
    "Very Low Calorie Diets (VLCDs) guidelines",
    "Intermittent Fasting methods and scientific evidence",
    "Time-Restricted Eating (TRE) benefits",
    "Mediterranean Diet for weight management and cardiovascular health",
    "Volumetrics Diet plan",
    "High Protein Diets for satiety and muscle preservation",
    "Dietary Fiber types and benefits for weight management",
    "dietary fiber and metabolic health", # New & specific
    "Meal Replacement Shakes for controlled calorie intake",
    "Exercise for weight loss and maintenance",
    "Aerobic exercise for fat loss",
    "aerobic exercise for metabolic health", # New & specific
    "Resistance training for weight management and body composition",
    "High-intensity interval training (HIIT) for fat loss", # New
    "NEAT (Non-Exercise Activity Thermogenesis) importance",
    "Behavioral Therapy for sustainable weight loss",
    "Comprehensive Lifestyle Intervention for obesity",
    "Cognitive Behavioral Therapy (CBT) for obesity and eating disorders",
    "Motivational Interviewing techniques for health behavior change",
    "Mindful Eating practices",
    "Stress management techniques for weight control", # Covers "stress management for weight control"
    "Cortisol effects on appetite and fat storage",
    "Sleep hygiene impact on weight and metabolism",
    "sleep quality and metabolic syndrome", # New
    "Weight cycling risks and management",
    "Childhood obesity prevention and treatment",
    "Brown Adipose Tissue (BAT) activation for weight loss", # Retained, lower priority
    "White Adipose Tissue (WAT) browning", # Retained, lower priority

    # Metabolic Disorders (Integrated with Obesity where relevant)
    "Metabolic Syndrome (MetS) diagnostic criteria and management",
    "Insulin Resistance mechanisms, diagnosis, and reversal",
    "Prediabetes diagnosis and intervention", # Existing, re-prioritized
    "prediabetes diagnosis and management", # New & specific
    "Type 2 Diabetes prevention, management, and remission",
    "Polycystic Ovary Syndrome (PCOS) metabolic features and treatment",
    "polycystic ovary syndrome (PCOS) and insulin resistance", # New & specific
    "Gestational Diabetes risks and management", # Existing, re-prioritized
    "gestational diabetes management", # New & specific
    "Non-alcoholic Fatty Liver Disease (NAFLD) and NASH progression",
    "Dyslipidemia diagnosis and lipid management",
    "Hypertriglyceridemia causes and treatment",
    "Low HDL cholesterol implications",
    "Hypertension relationship with metabolic syndrome", # Highly relevant to MetS
    "Glycemic Control strategies in diabetes",
    "HbA1c targets and interpretation",
    "Fasting Plasma Glucose test",
    "Oral Glucose Tolerance Test (OGTT) procedure",
    "Lipid Panel components and interpretation",
    "Cardiometabolic Health and risk factors",
    "Endothelial dysfunction in metabolic disorders",
    # Less central, but still relevant supporting terms for metabolic disorders
    "Type 1 Diabetes autoimmune mechanisms and treatment", # Retained for completeness on Diabetes
    "C-Reactive Protein (CRP) as an inflammatory marker", # General but relevant
    "Microvascular complications of diabetes (retinopathy, nephropathy, neuropathy)", # Supporting
    "Macrovascular complications of diabetes (heart disease, stroke)", # Supporting

    # Muscle Gain & Body Recomposition
    "Resistance Training principles for hypertrophy",
    "Progressive Overload in strength training",
    "Mechanical Tension as a driver for muscle growth",
    "Muscle Damage and repair processes in hypertrophy",
    "Metabolic Stress contribution to muscle growth",
    "Training Volume recommendations for muscle gain",
    "Training Frequency for optimal muscle protein synthesis",
    "Exercise Selection for targeted muscle development",
    "Periodization models for long-term strength and muscle gain",
    "Muscle Protein Synthesis (MPS) regulation",
    "Protein intake recommendations for muscle synthesis (grams per kg)",
    "Essential Amino Acids (EAAs) role in muscle building",
    "Leucine threshold for stimulating MPS",
    "Caloric Surplus for lean muscle mass gain",
    "Nutrient Timing strategies for muscle recovery and growth",
    "Carbohydrate requirements for strength athletes",
    "Dietary Fats role in hormone production and muscle health",
    "Creatine Monohydrate benefits, dosage, and safety for muscle growth",
    "Whey Protein types (isolate, concentrate, hydrolysate) and benefits",
    "Casein Protein for sustained amino acid release",
    "Beta-Alanine supplementation for performance and muscle endurance",
    "Citrulline Malate for improved blood flow and performance",
    "HMB (Beta-hydroxy beta-methylbutyrate) for muscle preservation",
    "Body recomposition strategies", # New
    "Sarcopenia diagnosis, treatment, and prevention", # Relevant to muscle health, esp. aging
    "Sarcopenic Obesity characteristics and management", # Intersection of core topics
    # Less central/more niche for muscle gain
    "Branched-Chain Amino Acids (BCAAs) supplementation efficacy", # Often debated, lower priority
    "Anabolic Steroids use, side effects, and health risks", # For completeness on muscle gain topics
    "Muscle dysmorphia symptoms and treatment", # Psychological aspect

    # General but Highly Relevant (Supporting Core Topics)
    "Macronutrient Balance for health and body composition",
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
        """You are an AI assistant specializing in health and wellness, tasked with creating informative question-and-answer pairs from medical and scientific texts.
        Your goal is to generate a single, clear question that the provided article text answers, along with a detailed and accurate answer based ONLY on the information within the text.

        **Primary Focus for Q&A Generation:**
        Prioritize questions and answers directly relevant to the following topics:
        1.  **Obesity and its Management:** Causes, health risks, lifestyle interventions (diet, exercise, behavior change), dietary strategies, and approved medical treatments for obesity.
        2.  **Metabolic Disorders:** Specifically Type 2 Diabetes, Metabolic Syndrome, and Polycystic Ovary Syndrome (PCOS), with an emphasis on their relationship to diet, physical activity, and body weight.
        3.  **Sustainable Weight Loss Strategies:** Evidence-based approaches for achieving and maintaining healthy weight loss.
        4.  **Principles of Muscle Gain:** Nutritional requirements (e.g., protein intake, caloric balance) and training principles (e.g., resistance exercise) for building muscle mass.

        **Content Guidelines:**
        *   **Evidence-Based:** Focus on extracting information that is factual and scientifically supported within the provided text.
        *   **Actionable (where appropriate):** If the text supports it, formulate Q&A that provides practical or actionable insights for the reader.
        *   **Informative Tone:** The question and answer should be clear, concise, and presented in an objective, informative tone suitable for a health-focused application. Avoid overly technical jargon where possible, or explain it if necessary for understanding.
        *   **Specificity:** Generate questions that target specific details in the text, rather than overly broad questions.

        **Output Format:**
        Your output MUST be a single, valid JSON object with exactly two keys: "question" and "answer". Do not include any other text, formatting, or explanations outside of this JSON structure.

        Example:
        {{
            "question": "What are the primary recommended lifestyle modifications for managing Type 2 Diabetes?",
            "answer": "The primary recommended lifestyle modifications for managing Type 2 Diabetes, according to the text, include adopting a balanced dietary plan with controlled carbohydrate intake, engaging in regular physical activity (at least 150 minutes of moderate-intensity aerobic exercise per week), achieving and maintaining a healthy body weight, and regular monitoring of blood glucose levels."
        }}

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
