import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# --- ROBUSTNESS: Perform startup checks before loading any other app components ---
def perform_startup_checks():
    """
    Checks for essential files and configurations before the main app runs.
    Returns True if all checks pass, otherwise displays an error and returns False.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # 1. Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "LLM_MODEL", "ENTREZ_EMAIL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(
            f"FATAL ERROR: The following required environment variables are missing in your .env file: "
            f"{', '.join(missing_vars)}\\n\\n"  # Note the escaped \n for the subtask
            "Please copy the .env.example file to a new .env file and fill in your credentials."
        )
        return False
        
    # 2. Check if the vector database has been created
    if not os.path.exists("db"):
        st.error(
            "FATAL ERROR: The vector database directory ('db') was not found. "
            "The knowledge base has not been created yet.\\n\\n" # Note the escaped \n for the subtask
            "Please run the ingestion script first from your terminal:\\n" # Note the escaped \n for the subtask
            "1. Add your PDF files to the 'data' folder.\\n" # Note the escaped \n for the subtask
            "2. Run the command: `python ingest.py`"
        )
        return False
        
    return True

# --- Main App Execution ---

# Set the page configuration first
st.set_page_config(page_title="Weight Management AI", page_icon="💪")

# Run the startup checks. If they fail, stop the app immediately.
if not perform_startup_checks():
    st.stop()

# Import the graph class only AFTER checks have passed to prevent import errors.
from src.graph import WeightManagementGraph

# Set the main title for the app
st.title("💪 Weight Management & Body Composition AI")

@st.cache_resource
def get_graph():
    """
    Initializes and caches the compiled LangGraph agent.
    The @st.cache_resource decorator ensures this complex object is created only once.
    """
    print("INFO: Initializing LangGraph agent...")
    graph_builder = WeightManagementGraph()
    return graph_builder.compile_graph()

def log_feedback(interaction: dict, feedback: str):
    """Logs the user's feedback on an answer to a file for later review."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "interaction": interaction
    }
    # Append the log entry as a new line in the JSONL file
    with open("feedback_for_review.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Initialize the main application graph
app = get_graph()

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an AI assistant specializing in weight management, obesity, muscle gain, and metabolic disorders like Type 2 Diabetes and Metabolic Syndrome. How can I help you today?"}
    ]

# --- UI Components ---

# Sidebar for optional user profile information
with st.sidebar:
    st.header("Your Profile (Optional)")
    st.info("Providing this information can help generate more relevant answers, but it is not required.")
    user_profile = {
        "age": st.text_input("Age", ""),
        "sex": st.selectbox("Sex", ["", "Male", "Female", "Prefer not to say"]),
        "goal": st.text_input("Primary Goal (e.g., lose fat, gain muscle)", "")
    }

# Display existing chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input loop
if prompt := st.chat_input("Ask about nutrition, exercise, weight management, or metabolic health..."):
    # Add user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare the input for the graph
            graph_input = {"query": prompt, "user_profile": user_profile}
            
            final_answer = "Sorry, an unexpected error occurred. Please try again later." # Default error message
            try:
                # Invoke the LangGraph agent to get a response
                response = app.invoke(graph_input)

                # Safely get the final answer from the response state
                # The .get() is good, but we'll ensure final_answer is set even if 'response' is None or not a dict.
                if response and isinstance(response, dict):
                    final_answer = response.get("final_answer", "Sorry, an error occurred while processing your request in the graph.")
                elif response: # If response is not a dict but some other truthy value
                    final_answer = str(response) # Convert to string as a fallback
                # If response is None, the default 'Sorry, an unexpected error occurred...' will be used.

            except Exception as e:
                st.error(f"An application error occurred: {e}") # Show a streamlit error message
                print(f"ERROR during app.invoke: {e}", file=sys.stderr) # Log to stderr for server-side logs
                # final_answer is already set to a generic error message
            
            st.markdown(final_answer)
    
    # Add the assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    # Store the details of this latest interaction for the feedback buttons
    st.session_state.last_interaction = {
        "query": prompt,
        "answer": final_answer,
        "user_profile": user_profile
    }

# Display feedback buttons only after the first user interaction has occurred
if "last_interaction" in st.session_state:
    st.markdown("---") # Visual separator
    col1, col2, _ = st.columns([1, 2, 5])
    with col1:
        if st.button("👍 Good"):
            log_feedback(st.session_state.last_interaction, "good")
            st.success("Feedback saved! Thank you.")
    with col2:
        if st.button("👎 Bad"):
            log_feedback(st.session_state.last_interaction, "bad")
            st.error("Feedback logged for review. Thank you.")
