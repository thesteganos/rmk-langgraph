import streamlit as st
import json
import os
from datetime import datetime
from src.graph import WeightManagementGraph

# --- FIX: Check for the existence of the database before initializing the app ---
DB_PATH = "db"
if not os.path.exists(DB_PATH):
    st.set_page_config(page_title="Error", page_icon="🚨")
    st.title("🚨 Database Not Found")
    st.error(
        "The vector database has not been created yet. "
        "Please run the ingestion script first from your terminal:\n\n"
        "1. Add your PDF files to the 'data' folder.\n"
        "2. Run the command: `python ingest.py`"
    )
    st.stop() # Stop the app from running further

st.set_page_config(page_title="Weight Management AI", page_icon="💪")
st.title("💪 Weight Management & Body Composition AI")

@st.cache_resource
def get_graph():
    """Caches the compiled graph to avoid re-initializing."""
    print("Initializing graph...")
    graph_builder = WeightManagementGraph()
    return graph_builder.compile_graph()

def log_feedback(interaction: dict, feedback: str):
    """Logs the interaction and feedback to a review file."""
    log_entry = {"timestamp": datetime.now().isoformat(), "feedback": feedback, "interaction": interaction}
    with open("feedback_for_review.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

app = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am an AI assistant specializing in weight management. How can I help you today?"}]

with st.sidebar:
    st.header("Your Profile (Optional)")
    user_profile = {
        "age": st.text_input("Age", ""),
        "sex": st.selectbox("Sex", ["", "Male", "Female", "Prefer not to say"]),
        "goal": st.text_input("Primary Goal", "")
    }

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about nutrition, exercise, or weight loss..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph_input = {"query": prompt, "user_profile": user_profile}
            response = app.invoke(graph_input)
            final_answer = response.get("final_answer", "Sorry, an error occurred.")
            st.markdown(final_answer)
    
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    # Store the interaction details for the feedback buttons
    st.session_state.last_interaction = {"query": prompt, "answer": final_answer, "user_profile": user_profile}

# --- FIX: Only show feedback buttons AFTER an interaction has occurred ---
if "last_interaction" in st.session_state:
    col1, col2, _ = st.columns([1, 2, 5])
    with col1:
        if st.button("👍 Good"):
            log_feedback(st.session_state.last_interaction, "good")
            st.success("Feedback saved!")
    with col2:
        if st.button("👎 Bad"):
            log_feedback(_st.session_state.last_interaction, "bad")
            st.error("Feedback logged.")
