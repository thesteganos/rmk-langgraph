from langchain.tools import Tool
from Bio import Entrez
import os

# Professional Entrez Configuration
# This identifies your script to NCBI and gives you higher rate limits.
entrez_email = os.getenv("ENTREZ_EMAIL")
entrez_api_key = os.getenv("ENTREZ_API_KEY")

if not entrez_email:
    print("WARNING: ENTREZ_EMAIL not set in .env file. Using a default email.")
    entrez_email = "anonymous.user@example.com"

Entrez.email = entrez_email

if entrez_api_key:
    print("INFO: Entrez API key found. Using higher rate limits.")
    Entrez.api_key = entrez_api_key
else:
    print("WARNING: ENTREZ_API_KEY not set. Using lower, default rate limits.")

def search_pubmed(query: str) -> str:
    """
    Searches PubMed for a given query and returns a summary of the top 3 results' abstracts.
    Useful for finding specific scientific literature, clinical trials, or meta-analyses.
    """
    try:
        print(f"---TOOL: Searching PubMed for: {query}---")
        handle = Entrez.esearch(db="pubmed", term=query, retmax="3", sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        if not id_list:
            return "No relevant articles found on PubMed for that query."
            
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()
        
        return abstracts
    except Exception as e:
        return f"An error occurred while searching PubMed: {e}"

# Create the LangChain tool
pubmed_tool = Tool(
    name="PubMedSearch",
    func=search_pubmed,
    description="Use this tool to find specific scientific or medical research papers, clinical trials, or meta-analyses from the PubMed database."
)
