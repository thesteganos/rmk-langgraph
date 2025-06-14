import os
import json
import sys # For sys.stderr
from typing import TypedDict, Literal
from dotenv import load_dotenv
from neo4j import GraphDatabase # Added Neo4j import

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Removed HuggingFaceEmbeddings import, will get from .utils
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from .utils import get_embedding_model # Added import for centralized model loading

# Import the custom tool from the tools file
from .tools import pubmed_tool
from google.ai.generativelanguage_v1beta.types import Tool as GenAITool # Added for Native Gemini Search

# Load environment variables from .env file
load_dotenv()
DB_PATH = "db"

# This defines the structure of the state that is passed between nodes in the graph.
class GraphState(TypedDict):
    query: str
    user_profile: dict
    query_type: Literal["foundational", "protocol", "hybrid", "unsafe"]
    documents: list
    web_results: list
    neo4j_results: list # Added neo4j_results
    final_answer: str
    disclaimer_needed: bool

class WeightManagementGraph:
    """
    This class orchestrates the entire adaptive RAG workflow using LangGraph.
    It initializes all necessary components (LLMs, tools, retriever) and defines
    the nodes and edges of the state machine graph.
    """
    def __init__(self):
        """Initializes models, tools, and the vector database retriever."""
        # --- Robust Startup Checks ---
        google_api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("LLM_MODEL")
        if not google_api_key:
            raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file. The application cannot start.")
        if not model_name:
            raise ValueError("FATAL ERROR: LLM_MODEL not defined in .env file. The application cannot start.")
        if not os.path.exists(DB_PATH):
             raise FileNotFoundError(f"FATAL ERROR: The database directory '{DB_PATH}' was not found. Please run ingest.py first.")

        print(f"INFO: Using single LLM_MODEL for all tasks: {model_name}")

        # --- Model Initialization ---
        # We initialize two instances to allow for different temperature settings.
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
        self.web_search_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=google_api_key)
        
        # --- Retriever Initialization ---
        print("INFO: Loading embedding model for retriever...")
        embeddings = get_embedding_model()
        if not embeddings:
            # This is a critical failure, the application cannot work without embeddings.
            raise ValueError("FATAL ERROR: Embedding model could not be loaded. Application cannot start.")
        print("INFO: Embedding model loaded successfully for retriever.")
        self.retriever = Chroma(persist_directory=DB_PATH, embedding_function=embeddings).as_retriever(search_kwargs={'k': 3})

        # --- Neo4j Driver Initialization ---
        self.neo4j_driver = None
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            print("Warning: Neo4j environment variables not fully set. Neo4j RAG integration will be disabled.", file=sys.stderr)
        else:
            try:
                self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                self.neo4j_driver.verify_connectivity()
                print("Successfully connected to Neo4j for RAG.")
            except Exception as e:
                print(f"Error: Could not connect to Neo4j for RAG: {e}. Neo4j RAG integration will be disabled.", file=sys.stderr)
                self.neo4j_driver = None
        
        # --- Tool & Agent Initialization ---
        print("INFO: Initializing tools...")
        # pubmed_tool is imported from .tools
        self.tools = [pubmed_tool]
        # self.web_search_llm will be used for native search in hybrid_rag_node
        # self.llm_with_tools is now specifically for LLM calls that might use pubmed_tool via Langchain's bind_tools mechanism.
        # If other nodes also need ONLY pubmed_tool, they can use this.
        # If a node needs NO tools or different tools, it should use self.llm or self.web_search_llm directly.
        self.llm_with_tools = self.web_search_llm.bind_tools(self.tools)
        print(f"INFO: LLM bound with custom tools: {[tool.name for tool in self.tools]}")

    ### NODE DEFINITIONS ###

    def safety_filter_node(self, state: GraphState) -> dict:
        """First node to check if the query is safe to answer."""
        node_name = "safety_filter_node"
        print(f"---NODE: {node_name}---")
        try:
            prompt = ChatPromptTemplate.from_template(
                """You are a safety classification system. Analyze the user's query.
                If the query asks for harmful, dangerous, or unethical advice related to weight loss, eating disorders, or steroid use, classify it as "unsafe".
                Otherwise, classify it as "safe". Respond with ONLY the word "safe" or "unsafe".
                Query: {query}"""
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": state["query"]})
            return {"query_type": "unsafe" if "unsafe" in result.lower() else None, "final_answer": ""}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "query_type": "unsafe", # Default to unsafe on error for safety
                "final_answer": f"Sorry, an unexpected error occurred in {node_name}.",
                "disclaimer_needed": False, # Ensure all state keys are present
                "documents": [],
                "web_results": [],
                "neo4j_results": [] # Added neo4j_results
            }

    def classify_query_node(self, state: GraphState) -> dict:
        """Classifies the query to determine the appropriate workflow path."""
        node_name = "classify_query_node"
        print(f"---NODE: {node_name}---")
        try:
            prompt = ChatPromptTemplate.from_template(
                """Classify the user's query into one of three categories:
                1. foundational: For basic scientific principles or definitions. (e.g., "What is a calorie?")
                2. protocol: For established, evidence-based practices or guidelines. (e.g., "What is the recommended protein intake?")
                3. hybrid: For trendy topics, supplements, user-specific situations, or very recent research. (e.g., "Is the carnivore diet good?")
                Respond with ONLY the category word. Query: {query}"""
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": state["query"]})
            classification = result.lower().strip()
            if classification not in ["foundational", "protocol", "hybrid"]:
                classification = "hybrid" # Default to hybrid if classification is unclear
            print(f"Classification result: {classification}")
            return {"query_type": classification, "final_answer": ""}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "query_type": "hybrid", # Default to hybrid on error
                "final_answer": f"Sorry, an unexpected error occurred in {node_name}.",
                "disclaimer_needed": False,
                "documents": [],
                "web_results": [],
                "neo4j_results": [] # Added neo4j_results
            }

    def foundational_node(self, state: GraphState) -> dict:
        """Answers simple questions directly from the LLM's memory."""
        node_name = "foundational_node"
        print(f"---NODE: {node_name}---")
        try:
            answer = self.llm.invoke(state["query"])
            # Pass through neo4j_results, even if empty
            return {"final_answer": answer.content, "disclaimer_needed": False, "documents": [], "web_results": [], "neo4j_results": state.get("neo4j_results", [])}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "final_answer": f"Sorry, an unexpected error occurred in {node_name}.",
                "disclaimer_needed": False,
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": [],
                "neo4j_results": [] # Added neo4j_results
            }

    def simple_neo4j_retriever(self, query: str) -> list[str]:
        if not self.neo4j_driver:
            return []

        keyword = query  # Using the whole query as keyword for simplicity

        cypher_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($keyword)
        OPTIONAL MATCH (e)-[r]-(related_node:Entity)
        WITH e, r, related_node
        LIMIT 20
        WITH e, collect({{name: related_node.name, type: related_node.type, rel_type: type(r), start_node_name: startNode(r).name, end_node_name: endNode(r).name}}) AS relations_details
        RETURN e.name AS entityName, e.type AS entityType, relations_details
        LIMIT 5
        """

        results_text_list = []
        try:
            with self.neo4j_driver.session() as session:
                records = session.run(cypher_query, keyword=keyword)
                for record in records:
                    entity_name = record.get("entityName")
                    entity_type = record.get("entityType")
                    relations_info = []
                    for rel_detail in record.get("relations_details", []):
                        if rel_detail and rel_detail['name'] and rel_detail['rel_type']:
                            rel_type = rel_detail['rel_type']
                            if rel_detail['start_node_name'] == entity_name:
                                direction = "-[" + rel_type + "]->"
                                connected_node_info = f"{rel_detail['name']}({rel_detail['type']})"
                            else:
                                direction = "<-[" + rel_type + "]-"
                                connected_node_info = f"{rel_detail['name']}({rel_detail['type']})"
                            relations_info.append(f"{entity_name} {direction} {connected_node_info}")

                    context_str = f"Entity: {entity_name} ({entity_type}). Related: {', '.join(relations_info) if relations_info else 'No direct relations found matching the criteria.'}"
                    results_text_list.append(context_str)
                if results_text_list:
                     print(f"Neo4j retriever found for keyword '{keyword}': {results_text_list}")
        except Exception as e:
            print(f"Error during Neo4j retrieval with keyword '{keyword}': {e}", file=sys.stderr)
            return []
        return results_text_list

    def protocol_rag_node(self, state: GraphState) -> dict:
        """Answers based on the trusted knowledge base (vector + graph) and detects knowledge gaps."""
        node_name = "protocol_rag_node"
        print(f"---NODE: {node_name}---")
        try:
            chroma_docs = self.retriever.invoke(state["query"])
            chroma_context_str = "\n\n".join(doc.page_content for doc in chroma_docs)

            neo4j_context_list = self.simple_neo4j_retriever(state["query"])
            neo4j_context_str = "\n\n".join(neo4j_context_list)

            if neo4j_context_str:
                combined_context = f"Knowledge Base Documents (from vector search):\n{chroma_context_str}\n\nKnowledge Graph Information (from graph database):\n{neo4j_context_str}"
            else:
                combined_context = f"Knowledge Base Documents (from vector search):\n{chroma_context_str}"

            prompt_template_str = """You are a specialist medical AI. Your task is to answer the user's question based *only* on the trusted information provided in the context.
The context may include documents from a knowledge base (vector search) and structured information from a knowledge graph.
- If the context contains relevant information, provide a clear, evidence-based answer. Synthesize information from both sources if applicable.
- If the context (from both vector and graph searches) is empty or irrelevant to the question, you MUST respond with the exact phrase: "KNOWLEDGE_GAP".
- Do not use any external knowledge. Be concise.
Context:
{context}
Question: {question}"""
            prompt = ChatPromptTemplate.from_template(prompt_template_str)

            llm_processing_chain = prompt | self.llm | StrOutputParser()
            answer = llm_processing_chain.invoke({"context": combined_context, "question": state["query"]})

            return {
                "final_answer": answer,
                "disclaimer_needed": False,
                "documents": chroma_docs,
                "neo4j_results": neo4j_context_list,
                "query_type": state.get("query_type"),
                "web_results": state.get("web_results", []) # Ensure web_results is passed through
            }
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "final_answer": f"Sorry, an error occurred while retrieving information in {node_name}.",
                "disclaimer_needed": False,
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": state.get("web_results", []),
                "neo4j_results": []
            }

    def hybrid_rag_node(self, state: GraphState) -> dict:
        """Uses web search or PubMed tools to answer complex or recent queries."""
        node_name = "hybrid_rag_node"
        print(f"---NODE: {node_name}---")
        try:
            print(f"---NODE: {node_name} using Native Gemini Search---")
            # Use self.web_search_llm as it's configured with a potentially different temperature for web search tasks.
            # The native Google Search tool is passed directly in the invoke method.
            try:
                # Ensure GenAITool is available in this scope (it should be imported at the top of the file)
                native_google_search = GenAITool(google_search={})
                # Check if GOOGLE_API_KEY is available, though it should be due to startup checks
                if not os.getenv("GOOGLE_API_KEY"):
                    print("ERROR in hybrid_rag_node: GOOGLE_API_KEY not found for native search.", file=sys.stderr)
                    # Fallback or error state needed here
                    return {
                        "final_answer": "Sorry, Google API Key not configured for web search.",
                        "disclaimer_needed": True,
                        "query_type": state.get("query_type"), "documents": [], "web_results": [], "neo4j_results": []
                    }

                answer_obj = self.web_search_llm.invoke(
                    state["query"],
                    tools=[native_google_search]
                )
                # The response structure for native tool usage might be different.
                # We need to extract the text content. If the model uses the tool, the response might include tool_calls.
                # If it answers directly after searching, it will be in answer_obj.content.
                # For now, assume direct content. If tool_calls are made and need handling, this logic would expand.
                final_answer_content = getattr(answer_obj, 'content', str(answer_obj))

                # If the answer_obj contains tool_calls, it means Gemini *wants* to call Google Search,
                # but for built-in tools, it often returns the search result directly in the content if it can.
                # The example `resp = llm.invoke("query", tools=[GenAITool(google_search={})])` shows `resp.content` having the answer.
                # We'll assume this behavior.

            except Exception as e:
                print(f"ERROR in {node_name} during native Gemini search: {e}", file=sys.stderr)
                # Ensure all state keys are present on error
                return {
                    "final_answer": f"Sorry, an error occurred during the web search process in {node_name}.",
                    "disclaimer_needed": True,
                    "query_type": state.get("query_type"), "documents": [], "web_results": [], "neo4j_results": []
                }

            # The rest of the return statement for this node:
            return {
                "final_answer": final_answer_content,
                "disclaimer_needed": True, # Web search was used
                "web_results": state.get("web_results", []), # This might need to be populated based on search if possible
                "documents": state.get("documents", []),
                "neo4j_results": state.get("neo4j_results", [])
            }
        except Exception as e: # This outer try-except is now redundant due to the inner one, but kept for safety.
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "final_answer": f"Sorry, an error occurred during the web search process in {node_name}.",
                "disclaimer_needed": True,
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": [],
                "neo4j_results": [] # Added neo4j_results
            }

    def log_and_reroute_node(self, state: GraphState) -> dict:
        """Logs the knowledge gap and prepares for a web search."""
        node_name = "log_and_reroute_node"
        print(f"---NODE: {node_name}---")
        try:
            # Ensure user_profile exists in state, default to empty dict if not
            user_profile = state.get("user_profile", {})
            with open("knowledge_gaps.jsonl", "a") as f:
                log_entry = {"query": state["query"], "user_profile": user_profile}
                f.write(json.dumps(log_entry) + "\n")
            return {"final_answer": state.get("final_answer","")} # Preserve existing final_answer if any
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            # This node doesn't typically set final_answer, but if it errors,
            # we should ensure the state is still valid.
            # It might be better to let the error propagate if logging is critical
            # or to ensure the graph can proceed to hybrid search anyway.
            # For now, just log and return empty or error state.
            return {
                "final_answer": f"Sorry, an error occurred during the logging process in {node_name}.",
                "query": state.get("query"),
                "user_profile": state.get("user_profile",{}),
                "query_type": state.get("query_type"),
                "documents": state.get("documents",[]),
                "web_results": state.get("web_results",[]),
                "disclaimer_needed": state.get("disclaimer_needed", False),
                "neo4j_results": state.get("neo4j_results", []) # Added neo4j_results
            }


    def add_disclaimer_node(self, state: GraphState) -> dict:
        """Adds a disclaimer to answers that used external tools."""
        print("---NODE: Add Disclaimer (if needed)---")
        if state.get("disclaimer_needed"):
            disclaimer = ("**Disclaimer:** This information may be based on web search results and is for informational purposes only. It is not a substitute for professional medical advice. Always seek the advice of your physician.")
            return {"final_answer": f"{disclaimer}\n\n---\n\n{state['final_answer']}"}
        return {}

    def canned_safety_response_node(self, state: GraphState) -> dict:
        """Returns a fixed, safe response for harmful queries."""
        print("---NODE: Canned Safety Response---")
        return {"final_answer": "I cannot answer this question as it seeks potentially harmful advice. Please consult with a qualified healthcare professional for any health-related concerns."}

    ### CONDITIONAL EDGE LOGIC ###

    def decide_after_safety(self, state: GraphState) -> str:
        """Routes to the main classifier or the safety response node."""
        return "canned_safety_response" if state["query_type"] == "unsafe" else "classify_query"

    def decide_after_classification(self, state: GraphState) -> str:
        """Routes to the appropriate workflow based on query type."""
        return state["query_type"]

    def decide_after_protocol_rag(self, state: GraphState) -> str:
        """Checks if a knowledge gap was found and decides whether to reroute."""
        # Check if final_answer exists and is a string before checking "KNOWLEDGE_GAP"
        final_answer = state.get("final_answer", "")
        if isinstance(final_answer, str) and "KNOWLEDGE_GAP" in final_answer:
            return "log_and_reroute"
        # If final_answer indicates an error occurred in a previous node,
        # it might be better to go straight to END or a dedicated error handling node.
        # For now, proceed to disclaimer if not a KNOWLEDGE_GAP.
        return "add_disclaimer"

    ### GRAPH COMPILATION ###

    def compile_graph(self):
        """Builds and compiles the complete LangGraph agent."""
        graph = StateGraph(GraphState)

        # Add all nodes to the graph
        graph.add_node("safety_filter", self.safety_filter_node)
        graph.add_node("classify_query", self.classify_query_node)
        graph.add_node("foundational", self.foundational_node)
        graph.add_node("protocol", self.protocol_rag_node)
        graph.add_node("hybrid", self.hybrid_rag_node)
        graph.add_node("log_and_reroute", self.log_and_reroute_node)
        graph.add_node("add_disclaimer", self.add_disclaimer_node)
        graph.add_node("canned_safety_response", self.canned_safety_response_node)

        # Set the entry point of the graph
        graph.set_entry_point("safety_filter")

        # Define the edges and conditional routing
        graph.add_conditional_edges(
            "safety_filter",
            self.decide_after_safety,
            {"canned_safety_response": "canned_safety_response", "classify_query": "classify_query"}
        )
        graph.add_conditional_edges(
            "classify_query",
            self.decide_after_classification,
            {"foundational": "foundational", "protocol": "protocol", "hybrid": "hybrid"}
        )
        graph.add_conditional_edges(
            "protocol",
            self.decide_after_protocol_rag,
            {"log_and_reroute": "log_and_reroute", "add_disclaimer": "add_disclaimer"}
        )
        
        graph.add_edge("log_and_reroute", "hybrid")
        graph.add_edge("foundational", "add_disclaimer")
        graph.add_edge("hybrid", "add_disclaimer")
        
        # Define the end points of the graph
        graph.add_edge("add_disclaimer", END)
        graph.add_edge("canned_safety_response", END)

        return graph.compile()
