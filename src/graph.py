import os
import json
import sys # For sys.stderr
from typing import TypedDict, Literal
from dotenv import load_dotenv

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
        
        # --- Tool & Agent Initialization ---
        self.tools = [
            self.web_search_llm.get_tools("google_search_retrieval")[0],
            pubmed_tool
        ]
        self.llm_with_tools = self.web_search_llm.bind_tools(self.tools)

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
                "web_results": []
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
                "web_results": []
            }

    def foundational_node(self, state: GraphState) -> dict:
        """Answers simple questions directly from the LLM's memory."""
        node_name = "foundational_node"
        print(f"---NODE: {node_name}---")
        try:
            answer = self.llm.invoke(state["query"])
            return {"final_answer": answer.content, "disclaimer_needed": False}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "final_answer": f"Sorry, an unexpected error occurred in {node_name}.",
                "disclaimer_needed": False,
                 # Ensure other relevant GraphState keys are present if needed by subsequent nodes
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": []
            }

    def protocol_rag_node(self, state: GraphState) -> dict:
        """Answers based on the trusted knowledge base and detects knowledge gaps."""
        node_name = "protocol_rag_node"
        print(f"---NODE: {node_name}---")
        try:
            prompt = ChatPromptTemplate.from_template(
                """You are a specialist medical AI. Your task is to answer the user's question based *only* on the trusted documents provided in the context.
                - If the context contains relevant information, provide a clear, evidence-based answer.
                - If the context is empty or irrelevant, you MUST respond with the exact phrase: "KNOWLEDGE_GAP".
                - Do not use any external knowledge.
                Context: {context}
                Question: {question}"""
            )
            rag_chain = (
                {"context": self.retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
                | prompt | self.llm | StrOutputParser()
            )
            answer = rag_chain.invoke(state["query"])
            # Ensure 'documents' key is populated if retriever runs, even if not explicitly used by this node's primary output
            # This might be more complex if retriever errors separately. For now, assume it runs before or within rag_chain.
            # If self.retriever itself can fail, that needs its own try-except.
            # For now, focusing on LLM call.
            return {"final_answer": answer, "disclaimer_needed": False, "documents": state.get("documents", [])}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            # If RAG fails, it's like a knowledge gap, so might reroute to hybrid or provide error.
            # For now, providing error. Could also return "KNOWLEDGE_GAP" to trigger rerouting.
            return {
                "final_answer": f"Sorry, an error occurred while retrieving information from the knowledge base in {node_name}.",
                "disclaimer_needed": False,
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": []
            }

    def hybrid_rag_node(self, state: GraphState) -> dict:
        """Uses web search or PubMed tools to answer complex or recent queries."""
        node_name = "hybrid_rag_node"
        print(f"---NODE: {node_name}---")
        try:
            answer = self.llm_with_tools.invoke(state["query"])
            # Assuming answer might be a complex object, ensure we get .content if applicable
            final_answer_content = getattr(answer, 'content', str(answer))
            return {"final_answer": final_answer_content, "disclaimer_needed": True, "web_results": state.get("web_results", [])}
        except Exception as e:
            print(f"ERROR in {node_name}: {e}", file=sys.stderr)
            return {
                "final_answer": f"Sorry, an error occurred during the web search process in {node_name}.",
                "disclaimer_needed": True, # Still good to have disclaimer if error happens during web search
                "query_type": state.get("query_type"),
                "documents": [],
                "web_results": []
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
                 # Keep other state elements as they are or default
                "query": state.get("query"),
                "user_profile": state.get("user_profile",{}),
                "query_type": state.get("query_type"),
                "documents": state.get("documents",[]),
                "web_results": state.get("web_results",[]),
                "disclaimer_needed": state.get("disclaimer_needed", False)
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
