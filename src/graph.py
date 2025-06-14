import os
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

from .tools import pubmed_tool

load_dotenv()
DB_PATH = "db"

class GraphState(TypedDict):
    query: str
    user_profile: dict
    query_type: Literal["foundational", "protocol", "hybrid", "unsafe"]
    documents: list
    web_results: list
    final_answer: str
    disclaimer_needed: bool

class WeightManagementGraph:
    """The core logic of the adaptive agent, implemented as a LangGraph."""
    def __init__(self):
        # --- FIX: More robust API key and dependency check ---
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file. The application cannot start.")
        if not os.path.exists(DB_PATH):
             raise FileNotFoundError(f"FATAL ERROR: The database directory '{DB_PATH}' was not found. Please run ingest.py first.")
            
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)
        self.web_search_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        self.retriever = Chroma(persist_directory=DB_PATH, embedding_function=embeddings).as_retriever(search_kwargs={'k': 3})
        
        self.tools = [
            self.web_search_llm.get_tools("google_search_retrieval")[0],
            pubmed_tool
        ]
        self.llm_with_tools = self.web_search_llm.bind_tools(self.tools)

    # ... (All other nodes and graph compilation logic remain the same as the previous "fixed" version) ...
    # ... (The robust prompt in protocol_rag_node is still correct) ...
    def safety_filter_node(self, state: GraphState) -> dict:
        print("---NODE: Safety Filter---")
        prompt = ChatPromptTemplate.from_template(
            """You are a safety classification system. Analyze the user's query.
            If the query asks for harmful, dangerous, or unethical advice related to weight loss, eating disorders, or steroid use, classify it as "unsafe".
            Otherwise, classify it as "safe". Respond with ONLY the word "safe" or "unsafe".
            Query: {query}"""
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": state["query"]})
        return {"query_type": "unsafe"} if "unsafe" in result.lower() else {"query_type": None}

    def classify_query_node(self, state: GraphState) -> dict:
        print("---NODE: Classify Query---")
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
            classification = "hybrid"
        print(f"Classification result: {classification}")
        return {"query_type": classification}

    def foundational_node(self, state: GraphState) -> dict:
        print("---NODE: Foundational Answer---")
        answer = self.llm.invoke(state["query"])
        return {"final_answer": answer.content, "disclaimer_needed": False}

    def protocol_rag_node(self, state: GraphState) -> dict:
        print("---NODE: Protocol RAG Answer---")
        
        prompt = ChatPromptTemplate.from_template(
            """You are a specialist medical AI. Your task is to answer the user's question based *only* on the trusted documents provided in the context.
            
            - If the context contains relevant information, provide a clear, evidence-based answer based on that information.
            - If the context is empty or does not contain information relevant to the question, you MUST state that you could not find an answer in the trusted knowledge base.
            - Do not use any external knowledge or your own memory.
            
            Context:
            {context}
            
            Question:
            {question}
            """
        )
        
        rag_chain = (
            {"context": self.retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
            | prompt | self.llm | StrOutputParser()
        )
        answer = rag_chain.invoke(state["query"])
        return {"final_answer": answer, "disclaimer_needed": False}

    def hybrid_rag_node(self, state: GraphState) -> dict:
        print("---NODE: Hybrid RAG (Multi-Tool)---")
        answer = self.llm_with_tools.invoke(state["query"])
        return {"final_answer": answer.content, "disclaimer_needed": True}

    def add_disclaimer_node(self, state: GraphState) -> dict:
        print("---NODE: Add Disclaimer (if needed)---")
        if state.get("disclaimer_needed"):
            disclaimer = ("**Disclaimer:** This information may be based on web search results and is for informational purposes only. It is not a substitute for professional medical advice. Always seek the advice of your physician.")
            return {"final_answer": f"{disclaimer}\n\n---\n\n{state['final_answer']}"}
        return {}

    def canned_safety_response_node(self, state: GraphState) -> dict:
        print("---NODE: Canned Safety Response---")
        return {"final_answer": "I cannot answer this question as it seeks potentially harmful advice. Please consult with a qualified healthcare professional for any health-related concerns."}

    def decide_after_safety(self, state: GraphState):
        return "canned_safety_response" if state["query_type"] == "unsafe" else "classify_query"

    def decide_after_classification(self, state: GraphState):
        return state["query_type"]

    def compile_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("safety_filter", self.safety_filter_node)
        graph.add_node("classify_query", self.classify_query_node)
        graph.add_node("foundational", self.foundational_node)
        graph.add_node("protocol", self.protocol_rag_node)
        graph.add_node("hybrid", self.hybrid_rag_node)
        graph.add_node("add_disclaimer", self.add_disclaimer_node)
        graph.add_node("canned_safety_response", self.canned_safety_response_node)
        graph.set_entry_point("safety_filter")
        graph.add_conditional_edges("safety_filter", self.decide_after_safety, {"canned_safety_response": "canned_safety_response", "classify_query": "classify_query"})
        graph.add_conditional_edges("classify_query", self.decide_after_classification, {"foundational": "foundational", "protocol": "protocol", "hybrid": "hybrid"})
        graph.add_edge("foundational", "add_disclaimer")
        graph.add_edge("protocol", "add_disclaimer")
        graph.add_edge("hybrid", "add_disclaimer")
        graph.add_edge("add_disclaimer", END)
        graph.add_edge("canned_safety_response", END)
        return graph.compile()
