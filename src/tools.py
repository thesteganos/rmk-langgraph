from langchain.tools import Tool
from Bio import Entrez
import os
import sys # Added for sys.stderr
from urllib.error import HTTPError # For specific network errors

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
    esearch_handle = None
    efetch_handle = None
    try:
        print(f"---TOOL: Searching PubMed for: {query}---")
        
        # ESearch part
        esearch_handle = Entrez.esearch(db="pubmed", term=query, retmax="3", sort="relevance")
        # Entrez.read can raise RuntimeError for parsing issues or if the handle is bad
        record = Entrez.read(esearch_handle)
        id_list = record["IdList"]

        if not id_list:
            return "No relevant articles found on PubMed for that query."
            
        # EFetch part
        efetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = efetch_handle.read() # This can also fail
        
        return abstracts

    except HTTPError as e:
        print(f"ERROR in PubMed search (HTTPError) for query '{query}': {e}", file=sys.stderr)
        return f"PubMed search failed due to a network connection error (Code: {e.code}). Please check your internet connection and ensure NCBI services are available."
    except IOError as e: # Catches network issues not covered by HTTPError, or local I/O problems
        print(f"ERROR in PubMed search (IOError) for query '{query}': {e}", file=sys.stderr)
        return "PubMed search failed due to a network or I/O issue. Please check your connection or try again later."
    except RuntimeError as e: # Often from Entrez.read() if XML is malformed or other Entrez lib errors
        print(f"ERROR in PubMed search (RuntimeError) for query '{query}': {e}", file=sys.stderr)
        return f"PubMed search failed while processing data from Entrez (e.g., parsing results). The error was: {e}"
    except Exception as e:
        error_type_name = type(e).__name__
        print(f"ERROR in PubMed search (Unexpected {error_type_name}) for query '{query}': {e}", file=sys.stderr)
        return f"An unexpected error ({error_type_name}) occurred while searching PubMed. Please try again later."
    finally:
        if esearch_handle:
            try:
                esearch_handle.close()
            except Exception as e:
                print(f"ERROR closing esearch_handle: {e}", file=sys.stderr)
        if efetch_handle:
            try:
                efetch_handle.close()
            except Exception as e:
                print(f"ERROR closing efetch_handle: {e}", file=sys.stderr)

# Create the LangChain tool
pubmed_tool = Tool(
    name="PubMedSearch",
    func=search_pubmed,
    description="Use this tool to find specific scientific or medical research papers, clinical trials, or meta-analyses from the PubMed database."
)
