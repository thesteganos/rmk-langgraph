import os
import sys # Moved sys import up
import json # Added json import
from neo4j import GraphDatabase # Added Neo4j import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate # Added ChatPromptTemplate import
from langchain_google_genai import ChatGoogleGenerativeAI # Added ChatGoogleGenerativeAI import
from dotenv import load_dotenv
from src.utils import get_embedding_model

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to 'python' to potentially mitigate protobuf issues
# This should be set before any libraries using protobuf are heavily utilized.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DATA_PATH = "data"
DB_PATH = "db"
PROCESSED_LOG_FILE = os.path.join(DB_PATH, "processed_files.log")

# --- Neo4j Helper Functions ---

def extract_graph_data_from_chunk(text_chunk, llm):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in biomedical and health knowledge graph generation.
        From the following text, extract entities and their relationships, focusing primarily on the areas detailed below.
        Return the output as a single, valid JSON object with two keys: "entities" and "relationships".

        **Primary Focus Areas & Entity Types:**

        1.  **Obesity & Weight Management:**
            *   Entities:
                *   `Condition_Obesity` (e.g., "Obesity", "Childhood Obesity", "Abdominal Obesity")
                *   `Factor_Obesity` (e.g., "Genetic predisposition", "Sedentary lifestyle", "High-calorie diet") - for causes/contributors
                *   `Complication_Obesity` (e.g., "Hypertension associated with obesity", "Sleep apnea")
                *   `Treatment_WeightManagement` (e.g., "Bariatric surgery", "Calorie restriction", "Semaglutide") - can include lifestyle, medication, surgery
                *   `Diet_WeightManagement` (e.g., "Ketogenic Diet", "Mediterranean Diet for weight loss")
        2.  **Metabolic Health:**
            *   Entities:
                *   `Condition_Metabolic` (e.g., "Type 2 Diabetes", "Metabolic Syndrome", "PCOS", "Insulin Resistance")
                *   `Biomarker_Metabolic` (e.g., "HbA1c", "Fasting Glucose", "Insulin Levels", "Homocysteine")
                *   `Process_Metabolic` (e.g., "Insulin signaling pathway", "Glycolysis", "Gluconeogenesis")
                *   `Treatment_Metabolic` (e.g., "Metformin", "Lifestyle modification for diabetes")
        3.  **Nutrition:**
            *   Entities:
                *   `Nutrient_Macro` (e.g., "Protein", "Carbohydrate", "Fat", "Saturated Fat")
                *   `Nutrient_Micro` (e.g., "Vitamin D", "Iron", "Omega-3 fatty acids")
                *   `DietaryPattern` (e.g., "High-protein diet", "Low-glycemic diet") - distinct from specific named diets for weight management unless context implies that.
                *   `Food` (e.g., "Avocado", "Salmon", "Leafy greens") - if specific foods are discussed with their properties.
        4.  **Exercise Physiology & Types:**
            *   Entities:
                *   `Exercise_Type` (e.g., "Resistance Training", "Aerobic Exercise", "HIIT")
                *   `Adaptation_Exercise` (e.g., "Muscle hypertrophy", "Increased VO2 max", "Mitochondrial biogenesis")
                *   `Principle_Training` (e.g., "Progressive Overload", "Specificity")
        5.  **Body Composition:**
            *   Entities:
                *   `Tissue_BodyComposition` (e.g., "Fat Mass", "Lean Body Mass", "Visceral Adipose Tissue")
                *   `Hormone_BodyComposition` (e.g., "Leptin", "Ghrelin", "Testosterone", "Cortisol") - specific to body comp regulation
                *   `Process_BodyComposition` (e.g., "Lipolysis", "Muscle protein synthesis", "Adipogenesis")

        **General Entity Types (Use if significant and not fitting above categories):**
            *   `Symptom` (e.g., "Fatigue")
            *   `Condition_General` (e.g., "Hypertension" - if not tied directly to obesity/metabolic syndrome in context)
            *   `Treatment_General` (e.g., "Aspirin")
            *   `Anatomy` (e.g., "Liver", "Pancreas", "Skeletal Muscle")
            *   `Gene` (e.g., "FTO gene")
            *   `Protein_General` (e.g., "Enzyme", "Receptor" - if not a specific hormone or biomarker above)
            *   `Molecular_Process` (e.g., "Inflammation", "Oxidative Stress" - if not covered by a more specific process type)
            *   `Concept_General` (e.g., "Public health", "Clinical trial")

        **Relationship Types:**
        Use descriptive relationship types (e.g., "CAUSES", "PREVENTS", "TREATS", "MANAGES", "INCREASES_RISK_OF", "DECREASES_RISK_OF", "ASSOCIATED_WITH", "PART_OF", "REGULATES", "PRODUCES", "INHIBITS", "STIMULATES", "MEASURES", "TYPE_OF", "SUBCLASS_OF", "IMPROVES", "WORSENS", "DIAGNOSES").
        Be specific. For example, instead of `{"source": "Metformin", "target": "Type 2 Diabetes", "type": "RELATED_TO"}`, prefer `{"source": "Metformin", "target": "Type 2 Diabetes", "type": "TREATS"}`.

        **Output Structure:**
        "entities" should be a list of objects, each with "name" (string) and "type" (string from the defined types).
        "relationships" should be a list of objects, each with "source" (string, entity name), "target" (string, entity name), and "type" (string).
        Ensure entity names in relationships match exactly those in the entities list.

        **Example (Illustrative - adapt types based on definitions above):**
        {{
            "entities": [
                {{"name": "Obesity", "type": "Condition_Obesity"}},
                {{"name": "Type 2 Diabetes", "type": "Condition_Metabolic"}},
                {{"name": "Sedentary Lifestyle", "type": "Factor_Obesity"}},
                {{"name": "Metformin", "type": "Treatment_Metabolic"}},
                {{"name": "Resistance Training", "type": "Exercise_Type"}},
                {{"name": "Muscle Hypertrophy", "type": "Adaptation_Exercise"}}
            ],
            "relationships": [
                {{"source": "Sedentary Lifestyle", "target": "Obesity", "type": "INCREASES_RISK_OF"}},
                {{"source": "Obesity", "target": "Type 2 Diabetes", "type": "INCREASES_RISK_OF"}},
                {{"source": "Metformin", "target": "Type 2 Diabetes", "type": "TREATS"}},
                {{"source": "Resistance Training", "target": "Muscle Hypertrophy", "type": "CAUSES"}}
            ]
        }}

        Prioritize comprehensive extraction based on the defined areas and types. If a concept is mentioned that fits a general type but is highly significant in the context of the primary focus areas, please include it.

        Text:
        {chunk}
        """
    )
    chain = prompt | llm
    try:
        response = chain.invoke({"chunk": text_chunk})
        content = response.content if hasattr(response, 'content') else str(response)
        cleaned_content = content.strip().replace("```json", "").replace("```", "").strip()
        graph_data = json.loads(cleaned_content)
        if "entities" not in graph_data or "relationships" not in graph_data:
            print(f"Warning: LLM output did not contain 'entities' or 'relationships' keys. Output: {cleaned_content}", file=sys.stderr)
            return None
        return graph_data
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON from LLM response for graph extraction: {e}. Response: {cleaned_content}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error during graph data extraction: {e}", file=sys.stderr)
        return None

def store_graph_data_in_neo4j(graph_data, driver):
    if not graph_data or not driver:
        return

    with driver.session() as session:
        for entity in graph_data.get("entities", []):
            if not entity.get("name") or not entity.get("type"):
                print(f"Skipping entity due to missing name or type: {entity}", file=sys.stderr)
                continue
            session.run(
                "MERGE (e:Entity {name: $name, type: $type})",
                name=entity["name"], type=entity["type"]
            )
        for rel in graph_data.get("relationships", []):
            if not rel.get("source") or not rel.get("target") or not rel.get("type"):
                print(f"Skipping relationship due to missing source, target, or type: {rel}", file=sys.stderr)
                continue
            session.run(
                """
                MATCH (source:Entity {{name: $source_name}})
                MATCH (target:Entity {{name: $target_name}})
                MERGE (source)-[r:{rel_type}]->(target)
                """.format(rel_type=rel["type"].upper().replace(" ", "_")), # Sanitize relationship type
                source_name=rel["source"], target_name=rel["target"]
            )
    print(f"Stored/merged {len(graph_data.get('entities',[]))} entities and {len(graph_data.get('relationships',[]))} relationships in Neo4j.")

# --- File Processing Functions ---

def get_processed_files():
    """Reads the log of already processed files."""
    try:
        with open(PROCESSED_LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def log_processed_file(filename):
    """Adds a new file to the processed log."""
    try:
        with open(PROCESSED_LOG_FILE, "a") as f:
            f.write(filename + "\n")
    except IOError as e:
        print(f"Error: Could not write to processed log file {PROCESSED_LOG_FILE}: {e}", file=sys.stderr)

def main():
    """Performs incremental ingestion of new PDF files into the ChromaDB vector store."""
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY"):
        print("Successfully loaded environment variables from .env file.")
    else:
        print("Info: .env file not found or key 'GOOGLE_API_KEY' is not set. Proceeding with environment variables.")
    print("--- Starting Incremental Ingestion Process ---")
    os.makedirs(DB_PATH, exist_ok=True)

    # --- Neo4j Connection Setup ---
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_driver = None
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("Warning: Neo4j environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) not fully set. Skipping Neo4j ingestion.", file=sys.stderr)
    else:
        try:
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            neo4j_driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Error: Could not connect to Neo4j: {e}. Skipping Neo4j ingestion.", file=sys.stderr)
            neo4j_driver = None
    # --- End Neo4j Connection Setup ---

    try:
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            print(f"The '{DATA_PATH}' directory is empty or does not exist. Please add PDF files to ingest.")
            if neo4j_driver:
                neo4j_driver.close()
                print("Neo4j connection closed.")
            return
    except OSError as e:
        print(f"Error: Could not access data directory {DATA_PATH}: {e}", file=sys.stderr)
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j connection closed.")
        return

    processed_files = get_processed_files()
    try:
        all_files = set(os.listdir(DATA_PATH))
    except OSError as e:
        print(f"Error: Could not list files in data directory {DATA_PATH}: {e}", file=sys.stderr)
        return
    new_files = sorted(list(all_files - processed_files))

    if not new_files:
        print("No new documents to process. Knowledge base is up-to-date.")
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j connection closed.")
        return

    print(f"Found {len(new_files)} new document(s) to process: {new_files}")

    all_new_documents = []
    successfully_loaded_files = [] # Keep track of files that loaded successfully

    print("Loading new documents one by one...")
    for filename in new_files:
        file_path = os.path.join(DATA_PATH, filename)
        try:
            print(f"Processing '{file_path}'...")
            loader = PyPDFLoader(file_path)
            documents_from_file = loader.load() # Returns a list of Document objects
            if documents_from_file:
                all_new_documents.extend(documents_from_file)
                successfully_loaded_files.append(filename) # Add to list of successfully loaded files
                print(f"Successfully loaded {len(documents_from_file)} document(s) from '{filename}'.")
            else:
                print(f"Warning: No documents found in '{filename}'. It might be empty or corrupted.", file=sys.stderr)
        except Exception as e: # Catching a broad exception as PyPDFLoader can raise various issues
            print(f"Error: Could not load or process PDF file {file_path}: {e}", file=sys.stderr)
            print(f"Skipping file {file_path} due to error.", file=sys.stderr)

    if not all_new_documents:
        print("No documents were successfully loaded from the new files. Nothing to add to the knowledge base.")
        # No need to log processed files if none were successful
        print("-----------------------------------------")
        print("Incremental ingestion complete (no new data added).")
        print("-----------------------------------------")
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j connection closed.")
        return

    print(f"Total of {len(all_new_documents)} new document(s) loaded from {len(successfully_loaded_files)} file(s).")

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_new_documents) # Use all_new_documents here
        print(f"Split into {len(texts)} chunks.")

        # --- Neo4j Graph Extraction ---
        llm_for_graph_extraction = None
        if neo4j_driver: # Only initialize LLM if Neo4j is available
            google_api_key = os.getenv("GOOGLE_API_KEY")
            model_name = os.getenv("LLM_MODEL") # Or a specific model for extraction
            if not google_api_key or not model_name:
                print("Warning: GOOGLE_API_KEY or LLM_MODEL not set. Cannot perform Neo4j graph extraction.", file=sys.stderr)
            else:
                try:
                    llm_for_graph_extraction = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
                    print("LLM for graph extraction initialized.")
                except Exception as e:
                    print(f"Error initializing LLM for graph extraction: {e}", file=sys.stderr)
                    llm_for_graph_extraction = None

        if neo4j_driver and llm_for_graph_extraction:
            print("Starting Neo4j graph data extraction and storage...")
            for i, doc_chunk in enumerate(texts): # texts are LangChain Document objects
                print(f"Extracting graph data from chunk {i+1}/{len(texts)}...")
                graph_data = extract_graph_data_from_chunk(doc_chunk.page_content, llm_for_graph_extraction)
                if graph_data:
                    store_graph_data_in_neo4j(graph_data, neo4j_driver)
            print("Neo4j graph data extraction and storage complete.")
        elif neo4j_driver and not llm_for_graph_extraction:
            print("Neo4j driver is available, but LLM for graph extraction is not. Skipping Neo4j processing.", file=sys.stderr)
        # --- End Neo4j Graph Extraction ---

        print("Loading embedding model...")
        embeddings = get_embedding_model()
        if not embeddings:
            # The error is already printed by get_embedding_model,
            # but we should not proceed if the model failed to load.
            print("Halting ingestion due to embedding model loading failure.", file=sys.stderr)
            if neo4j_driver:
                neo4j_driver.close()
                print("Neo4j connection closed.")
            return # Exit main()
        print("Embeddings model loaded successfully.")

        print("Adding new documents to the vector store...")
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        db.add_documents(texts) # Add the processed texts
        print("Knowledge base updated successfully.")

        # Log only the successfully processed files
        for filename in successfully_loaded_files:
            log_processed_file(filename)
        print(f"Updated the processed files log for {len(successfully_loaded_files)} file(s).")

    except Exception as e:
        # This is a general catch-all for errors during splitting, embedding, or DB add.
        # If this block is hit, none of the files from this run should be logged as processed.
        print(f"An error occurred during text processing or database update: {e}", file=sys.stderr)
        print("No files from this run will be marked as processed due to the error.", file=sys.stderr)
        # Note: successfully_loaded_files are not logged if this overall try block fails.

    print("-----------------------------------------")
    print("Incremental ingestion complete!")
    print("-----------------------------------------")

    if neo4j_driver:
        neo4j_driver.close()
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()
