# Main application file for Lighthouse HealthConnect: app.py
import streamlit as st
import os
import base64
import logging
import subprocess
import tempfile
from st_audiorec import st_audiorec

from modules.utils import get_file_hash, load_last_kb_hash, save_last_kb_hash
from modules.llm_handler import get_response
from modules.vector_store_manager import get_vector_store
from modules.audio_handler import speech_to_text

# --- Page Config ---
st.set_page_config(layout="wide", page_icon="assets/image.png", page_title="Lighthouse HealthConnect")

# --- Helper function to encode image ---
def get_image_as_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Caching & KB Update Logic (runs once on startup) ---
@st.cache_resource
def initialize_app_resources():
    """
    Initializes and caches heavy resources. Also handles KB updates.
    This function runs once and is cached by Streamlit.
    """
    data_dir = "data"
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    if zip_files:
        zip_path = os.path.join(data_dir, zip_files[0])
        current_hash = get_file_hash(zip_path)
        previous_hash = load_last_kb_hash()
        if current_hash != previous_hash:
            with st.spinner("New knowledge base detected. Updating... This may take a moment."):
                try:
                    subprocess.run(["python", "populate_kb.py", zip_path], check=True, capture_output=True, text=True)
                    save_last_kb_hash(current_hash)
                    st.toast("Knowledge base updated successfully! 🎉")
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to update knowledge base: {e.stderr}")
                    st.stop()
    return get_vector_store()

vector_store = initialize_app_resources()

# --- UI: Title and Layout ---
img_base64 = get_image_as_base64("assets/image.png")
if img_base64:
    st.markdown(f"<h1><img src='data:image/png;base64,{img_base64}' style='height: 1.2em; vertical-align: middle;'> Lighthouse HealthConnect 💬</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1>Lighthouse HealthConnect 💬</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Initialize Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Tools (Voice Recorder) ---
with st.sidebar:
    st.header("Tools")
    st.markdown("Record your question:")
    wav_audio_data = st_audiorec()

# --- Main Chat Area ---
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source"):
             st.caption(f"Source: {message['source']}")

# --- Handle User Input (from Voice or Text) ---
prompt = None

# 1. Process voice input if available
if wav_audio_data is not None:
    with st.spinner("Transcribing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(wav_audio_data)
            temp_audio_path = fp.name
        
        transcribed_text = speech_to_text(temp_audio_path)
        os.remove(temp_audio_path)
        
        if "Could not" not in transcribed_text and transcribed_text.strip():
            prompt = transcribed_text
            st.toast("Voice query transcribed!")
        else:
            st.error("Sorry, I couldn't understand the audio. Please try again.")

# 2. Process text input
if text_prompt := st.chat_input("Ask a question about Tuberculosis..."):
    prompt = text_prompt

# --- Main Logic: If there is a prompt, process it ---
if prompt:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if vector_store:
                result = get_response(prompt, vector_store)
                answer = result.get("answer", "Sorry, I encountered an error.")
                source = result.get("source", "N/A").split(" (Sources:")[0]
            else:
                answer = "Vector store not available. Please check the configuration."
                source = "Error"
            
            st.markdown(answer)
            st.caption(f"Source: {source}")

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer, "source": source})
    
    # Rerun to reflect the new message and clear input
    st.rerun()