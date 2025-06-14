# Connecting External Agents to the Weight Management AI

This document outlines how to connect external applications or "agents" to the Weight Management AI system, which leverages a sophisticated architecture combining LangGraph, hybrid Retrieval Augmented Generation (RAG), and a Knowledge Graph (KG).

## System Architecture Overview

The Weight Management AI relies on a system referred to as "rmk-langgraph hybrid RAG + KG". Here's a breakdown of its key components:

*   **LangGraph**: The core of the AI is built using [LangGraph](https://python.langchain.com/docs/langgraph/), a library for building stateful, multi-actor applications with LLMs. It orchestrates the entire query processing workflow, deciding which tools or knowledge sources to use based on the nature of the query.

*   **Hybrid Retrieval Augmented Generation (RAG)**: When a query requires factual information from its knowledge base, the system uses a hybrid RAG approach:
    *   **Chroma Vector Database**: A primary source for RAG is a ChromaDB vector store. This database is populated with processed information extracted from various documents (e.g., scientific articles, guidelines).
    *   **Neo4j Knowledge Graph (KG)**: To provide more structured and interconnected information, the system also queries a Neo4j graph database. This KG stores entities (like "metformin", "Type 2 Diabetes") and their relationships, allowing for more nuanced information retrieval.
    The information retrieved from both Chroma and Neo4j is synthesized by an LLM to generate a comprehensive answer.

*   **Integrated Tools**: For queries that require up-to-date information or access to specialized databases, the system can utilize:
    *   **Google Search**: For general web searches on new or trendy topics.
    *   **PubMed Search**: A dedicated tool to search the PubMed database for scientific and medical literature.

*   **Knowledge Base Population**:
    *   The `knowledge_pipeline.py` script is responsible for gathering and pre-processing information from trusted web sources and scientific literature. It generates question-answer pairs that are then reviewed.
    *   The `ingest.py` script (details to be confirmed, but typically) takes the reviewed data (e.g., from `pending_review.jsonl`) and populates the Chroma vector database and potentially the Neo4j knowledge graph. Regularly running these scripts ensures the AI's knowledge base remains current.

## Connecting Your Agent

External agents or applications can interact with the Weight Management AI by programmatically invoking its compiled LangGraph. The primary interface for this is the `WeightManagementGraph` class located in `src/graph.py`.

### Prerequisites

Before your agent can connect, ensure the following:

1.  **Environment Variables**: The system relies on several environment variables. A `.env.example` file is provided in the repository. Copy it to a `.env` file and fill in the necessary values, especially:
    *   `GOOGLE_API_KEY`: For Google Generative AI models and Google Search.
    *   `LLM_MODEL`: Specifies the generative model to be used.
    *   `ENTREZ_EMAIL` (and optionally `ENTREZ_API_KEY`): For PubMed searches.
    *   `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: For connecting to the Neo4j knowledge graph.
    *   Ensure all other variables as defined in `.env.example` are correctly set up.

2.  **Knowledge Base Ingestion**: The AI's effectiveness heavily depends on its knowledge base. Make sure you have run the ingestion process:
    *   Populate the `data/` folder with your source PDF documents.
    *   Run `python ingest.py` (or the equivalent script that processes `pending_review.jsonl` and populates ChromaDB and Neo4j). This step is crucial for the RAG components to function correctly. If the vector database directory (`db/`) or Neo4j data is missing, the system will likely fail or provide incomplete answers.

3.  **Python Environment**: Your agent will need access to the project's Python environment with all dependencies listed in `requirements.txt` installed.

### Connection Steps

Here's how your Python agent can connect to and use the Weight Management AI:

1.  **Import `WeightManagementGraph`**:
    ```python
    from src.graph import WeightManagementGraph
    ```

2.  **Initialize `WeightManagementGraph`**:
    This step loads the LLMs, retriever, tools, and sets up the Neo4j connection.
    ```python
    # Ensure .env variables are loaded, or use python-dotenv explicitly if needed
    # from dotenv import load_dotenv
    # load_dotenv()

    try:
        graph_builder = WeightManagementGraph()
    except Exception as e:
        print(f"Error initializing WeightManagementGraph: {e}")
        # Handle initialization errors (e.g., missing .env variables, DB path)
        exit()
    ```

3.  **Compile the Graph**:
    The `compile_graph()` method builds the LangGraph runnable agent.
    ```python
    compiled_agent = graph_builder.compile_graph()
    ```
    This `compiled_agent` is what you'll interact with. It's a good idea to cache this object if your agent handles multiple requests, as compilation can take a moment.

4.  **Invoke the Agent**:
    Pass a dictionary containing the user's query and profile to the agent's `invoke()` method.
    ```python
    user_query = "What are the benefits of resistance training for older adults?"
    user_profile_data = {
        "age": "65",
        "sex": "Female",
        "goal": "Maintain muscle mass"
    }

    input_data = {
        "query": user_query,
        "user_profile": user_profile_data
        # Other GraphState keys will be populated by the graph itself
    }

    try:
        response_state = compiled_agent.invoke(input_data)
        final_answer = response_state.get("final_answer", "No answer provided.")
        print(f"AI's Answer: {final_answer}")

        # The full response_state dictionary contains more details about the run
        # print(f"Full response state: {response_state}")

    except Exception as e:
        print(f"Error invoking agent: {e}")
        # Handle invocation errors
    ```

### Example Python Snippet

Here's a concise example putting it all together:

```python
import os
from dotenv import load_dotenv
from src.graph import WeightManagementGraph # Assuming your script is in the root or src is in PYTHONPATH

def get_ai_response(query: str, profile: dict = None) -> str:
    """
    Initializes, compiles, and invokes the Weight Management AI graph.
    Returns the final answer string.
    """
    load_dotenv() # Load environment variables from .env

    # Basic check for a critical environment variable
    if not os.getenv("GOOGLE_API_KEY"):
        return "Error: GOOGLE_API_KEY not found. Please set it in your .env file."

    if not os.path.exists("db"): # Check for vector DB
        return "Error: Vector database ('db') not found. Please run ingestion scripts."

    try:
        graph_builder = WeightManagementGraph()
        compiled_agent = graph_builder.compile_graph()

        input_data = {
            "query": query,
            "user_profile": profile if profile else {}
        }

        response_state = compiled_agent.invoke(input_data)
        return response_state.get("final_answer", "Error: No final answer in response.")

    except FileNotFoundError as fnf_error:
        # Specific handling for missing DB if not caught by WeightManagementGraph init
        return f"Error: A required file or directory was not found: {fnf_error}. Ensure knowledge base is ingested."
    except ValueError as val_error:
        # Catch configuration errors from WeightManagementGraph init
        return f"Configuration Error: {val_error}. Please check your .env file and setup."
    except Exception as e:
        # General error handling
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    test_query = "What is sarcopenia and how can it be managed?"
    test_profile = {"age": "70", "sex": "Male", "goal": "Improve strength"}

    answer = get_ai_response(test_query, test_profile)
    print(f"Query: {test_query}")
    print(f"AI Response: {answer}")

    test_query_simple = "What is a calorie?"
    simple_answer = get_ai_response(test_query_simple)
    print(f"Query: {test_query_simple}")
    print(f"AI Response: {simple_answer}")
```

## Input and Output Structure

Understanding the expected input and the structure of the output is crucial for interacting with the agent.

### Input to `invoke()`

The `compiled_agent.invoke()` method expects a single dictionary argument. This dictionary should conform to the initial state of the `GraphState` TypedDict (defined in `src/graph.py`). The key fields you need to provide are:

*   **`query` (str)**: This is the mandatory user query that you want the AI to answer.
*   **`user_profile` (dict)**: This is an optional dictionary containing user-specific information that can help the AI tailor its response. Common keys might include:
    *   `"age"` (str or int)
    *   `"sex"` (str, e.g., "Male", "Female")
    *   `"goal"` (str, e.g., "lose fat", "gain muscle")
    *   Any other relevant health or demographic data.
    If no user profile information is available, you can pass an empty dictionary `{}`.

Example input:
```python
input_data = {
    "query": "How much protein should I eat to build muscle?",
    "user_profile": {"age": "30", "sex": "Male", "goal": "Build muscle", "weight_kg": "75"}
}
```

### Output from `invoke()`

The `invoke()` method returns a dictionary which represents the final state of the graph's execution. This dictionary is an instance of the `GraphState` TypedDict. Key fields in the output include:

*   **`query` (str)**: The original query that was processed.
*   **`user_profile` (dict)**: The user profile that was used.
*   **`query_type` (Literal["foundational", "protocol", "hybrid", "unsafe"])**: The classification determined by the agent for the query. This can give insight into how the query was processed.
*   **`documents` (list)**: A list of documents retrieved from the Chroma vector database if the "protocol" RAG path was taken. Each item in the list is typically a LangChain `Document` object.
*   **`web_results` (list)**: A list of results obtained from web searches if the "hybrid" path involving Google Search was taken. The structure of these results may vary.
*   **`neo4j_results` (list)**: A list of strings, where each string is a textual representation of information retrieved from the Neo4j knowledge graph if the "protocol" RAG path (with Neo4j integration) was taken.
*   **`final_answer` (str)**: This is the primary output you'll likely be interested in. It contains the AI's textual answer to the query. If an error occurred or the query was deemed unsafe, this field will contain an appropriate message.
*   **`disclaimer_needed` (bool)**: A boolean indicating whether a disclaimer should be shown with the answer (e.g., if web search was used). The `add_disclaimer_node` in the graph automatically prepends a standard disclaimer to `final_answer` if this is true.

Example of accessing the final answer:
```python
response_state = compiled_agent.invoke(input_data)
answer = response_state.get("final_answer")

# You can also inspect other parts of the state:
# if response_state.get("disclaimer_needed"):
#     print("Note: This answer may include information from external web searches.")
# if response_state.get("documents"):
#     print(f"Retrieved {len(response_state['documents'])} documents from knowledge base.")
```

By inspecting these fields, your connecting agent can not only get the answer but also understand more about the AI's process and the sources it used.

## Understanding Key System Components (Optional)

While not strictly necessary for basic connection, understanding a bit more about the internal components can be helpful for advanced use cases or troubleshooting.

### Core Graph Components (`src/graph.py`)

The `WeightManagementGraph` orchestrates the query-answering process through several specialized nodes:

*   **`safety_filter_node`**: The first point of contact. It checks if a query is potentially harmful or unethical. If so, it routes to a canned, safe response.
*   **`classify_query_node`**: If the query is safe, this node classifies it into:
    *   `foundational`: Basic questions the LLM can answer directly.
    *   `protocol`: Questions requiring established, evidence-based information from the internal knowledge base (Chroma DB + Neo4j KG).
    *   `hybrid`: Trendy topics, new research, or user-specific queries that might require web searches or PubMed lookups.
*   **`foundational_node`**: Directly answers simple queries using the LLM's general knowledge.
*   **`protocol_rag_node`**: Implements the hybrid RAG approach. It fetches relevant information from:
    *   The **Chroma vector database** (for general text-based knowledge).
    *   The **Neo4j knowledge graph** (for structured entity and relationship data, via `simple_neo4j_retriever`).
    The combined context is then used by an LLM to generate an answer. If no relevant information is found in these trusted sources, it flags a "KNOWLEDGE_GAP".
*   **`hybrid_rag_node`**: Handles "hybrid" queries by:
    *   Utilizing **Google Search** (via native Gemini tool integration) for current events or general web knowledge.
    *   It can also be configured to use specific tools like the `pubmed_tool` (from `src/tools.py`) for accessing scientific literature, although the example `hybrid_rag_node` primarily shows native Google Search.
*   **`log_and_reroute_node`**: If the `protocol_rag_node` signals a "KNOWLEDGE_GAP", this node logs the gap (for future knowledge base improvement) and typically reroutes the query to the `hybrid_rag_node` to attempt an answer using web search.
*   **`add_disclaimer_node`**: If a query was answered using external sources (like web search), this node prepends a standard disclaimer to the `final_answer`.
*   **`canned_safety_response_node`**: Provides a generic, safe response if the initial `safety_filter_node` deems the query unsafe.

### Knowledge Base Customization and Updates

The AI's ability to answer "protocol" type questions accurately depends heavily on the content and currency of its knowledge base (Chroma DB and Neo4j KG).

*   **`knowledge_pipeline.py`**: This script is designed to be run periodically or as needed. It:
    *   Scrapes content from predefined `TRUSTED_SOURCES` and from web searches based on `TERMS_OF_INTEREST` using `TRUSTED_SEARCH_SITES`.
    *   Processes the fetched article text using an LLM to generate question-answer "propositions".
    *   Saves these propositions to `pending_review.jsonl`.
    *   **It is crucial to manually review `pending_review.jsonl` for accuracy and relevance before these propositions are added to the main knowledge base.**

*   **`ingest.py`** (Conceptual - actual implementation might vary):
    *   This script is responsible for taking the (ideally reviewed) data from `pending_review.jsonl` or other curated sources.
    *   It then processes this data, generates embeddings, and stores it in the **Chroma vector database** (`db/` directory).
    *   It would also be responsible for updating the **Neo4j knowledge graph** with new entities and relationships if the ingested data contains structured information suitable for the KG.
    *   Regularly running `knowledge_pipeline.py` followed by a review and then `ingest.py` is key to expanding and updating the AI's knowledge.

By customizing the sources in `knowledge_pipeline.py` and ensuring a robust review and ingestion process, you can tailor the AI's expertise to specific domains or keep it updated with the latest information.
