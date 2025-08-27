# app.py: Fixed main application file for Lighthouse HealthConnect
import sys
print(">>> Using Python executable:", sys.executable)

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
from modules.audio_handler import speech_to_text, text_to_speech, cleanup_audio_file

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_icon="🏥", 
    page_title="Lighthouse HealthConnect"
)

# --- Helper function to encode image ---
def get_image_as_base64(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception as e:
            st.warning(f"Could not load image {file_path}: {e}")
            return None
    return None

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Create data directory if it doesn't exist ---
if not os.path.exists("data"):
    os.makedirs("data")
    logger.info("Created data directory")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio_processed" not in st.session_state:
    st.session_state.last_audio_processed = None

if "audio_responses" not in st.session_state:
    st.session_state.audio_responses = {}

# --- Caching & KB Update Logic (runs once on startup) ---
@st.cache_resource
def initialize_app_resources():
    """
    Initializes and caches heavy resources. Also handles KB updates.
    This function runs once and is cached by Streamlit.
    """
    try:
        data_dir = "data"
        
        # Check if data directory exists and has ZIP files
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory '{data_dir}' not found. Creating it...")
            os.makedirs(data_dir)
            st.warning("No knowledge base found. Please add a ZIP file to the 'data' directory.")
            return None
            
        zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
        
        if not zip_files:
            st.warning("No ZIP files found in the 'data' directory. Please add your knowledge base ZIP file.")
            logger.warning("No ZIP files found in data directory")
            return None
        
        # Use the first ZIP file found
        zip_path = os.path.join(data_dir, zip_files[0])
        logger.info(f"Found knowledge base: {zip_path}")
        
        # Check if we need to update the knowledge base
        current_hash = get_file_hash(zip_path)
        previous_hash = load_last_kb_hash()
        
        if current_hash != previous_hash:
            st.info("🔄 New knowledge base detected. Updating... This may take a moment.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Running knowledge base update...")
                progress_bar.progress(25)
                
                # Run the populate_kb.py script
                result = subprocess.run(
                    [sys.executable, "populate_kb.py", zip_path], 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    cwd=os.getcwd()
                )
                
                progress_bar.progress(75)
                status_text.text("Saving update records...")
                
                # Save the new hash
                save_last_kb_hash(current_hash)
                
                progress_bar.progress(100)
                status_text.text("Knowledge base updated successfully!")
                
                st.success("✅ Knowledge base updated successfully!")
                logger.info("Knowledge base update completed successfully")
                
                # Clear the progress indicators after a moment
                import time
                time.sleep(2)
                progress_bar.empty()
                status_text.empty()
                
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Failed to update knowledge base:")
                st.code(e.stderr)
                logger.error(f"KB update failed: {e.stderr}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error during knowledge base update: {str(e)}")
                logger.error(f"Unexpected error during KB update: {e}")
                st.stop()
        
        # Get the vector store
        try:
            vector_store = get_vector_store()
            logger.info("Vector store initialized successfully")
            return vector_store
        except Exception as e:
            st.error(f"❌ Failed to initialize vector store: {str(e)}")
            logger.error(f"Vector store initialization failed: {e}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error initializing app resources: {str(e)}")
        logger.error(f"App initialization error: {e}")
        return None

# Initialize resources
with st.spinner("🚀 Initializing Lighthouse HealthConnect..."):
    vector_store = initialize_app_resources()

# --- UI: Title and Layout ---
img_base64 = get_image_as_base64("assets/image.png")
if img_base64:
    st.markdown(
        f"<h1><img src='data:image/png;base64,{img_base64}' style='height: 1.2em; vertical-align: middle;'> Lighthouse HealthConnect 💬</h1>", 
        unsafe_allow_html=True
    )
else:
    st.markdown("<h1>🏥 Lighthouse HealthConnect 💬</h1>", unsafe_allow_html=True)

st.markdown("---")

# Show status based on vector store availability
if vector_store is None:
    st.error("⚠️ Knowledge base not available. Please check the configuration and ensure you have a ZIP file in the 'data' directory.")
    st.info("💡 To use this app, please add your knowledge base ZIP file to the 'data' directory and restart the application.")
    st.stop()
else:
    st.success("✅ Knowledge base loaded and ready!")

# --- Sidebar for Tools (Voice Recorder) ---
with st.sidebar:
    st.header("🎤 Voice Tools")
    st.markdown("**Record your question:**")
    
    try:
        wav_audio_data = st_audiorec()
    except Exception as e:
        st.error(f"Audio recorder not available: {e}")
        wav_audio_data = None
    
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Speak clearly into your microphone")
    st.markdown("- Ask questions about tuberculosis")
    st.markdown("- You can ask in English, Yoruba, Igbo, Hausa, or Pidgin")
    st.markdown("- Voice responses are provided for voice queries")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_audio_processed = None
        st.session_state.audio_responses = {}
        st.rerun()

# --- Main Chat Area ---
st.markdown("### 💬 Chat")

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("source"):
            st.caption(f"📚 Source: {message['source']}")
        
        # Display audio player for voice responses (only for assistant messages with audio)
        if message["role"] == "assistant" and message.get("has_audio"):
            audio_key = f"audio_{idx}"
            if audio_key in st.session_state.audio_responses:
                try:
                    audio_file = st.session_state.audio_responses[audio_key]
                    if os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/mp3")
                except Exception as e:
                    logger.warning(f"Could not display audio for message {idx}: {e}")

# --- Handle User Input (from Voice or Text) ---
prompt = None
is_voice_input = False

# 1. Process voice input if available
if wav_audio_data is not None:
    # Create a hash of the audio data to check if it's new
    import hashlib
    audio_hash = hashlib.md5(wav_audio_data).hexdigest()
    
    # Only process if this is new audio (prevents re-processing)
    if audio_hash != st.session_state.last_audio_processed:
        st.session_state.last_audio_processed = audio_hash
        
        with st.spinner("🎧 Transcribing audio..."):
            try:
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                    fp.write(wav_audio_data)
                    temp_audio_path = fp.name
                
                # Transcribe audio
                transcribed_text = speech_to_text(temp_audio_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass  # Ignore cleanup errors
                
                # Check if transcription was successful
                if (transcribed_text and 
                    "Could not" not in transcribed_text and 
                    "Error" not in transcribed_text and 
                    len(transcribed_text.strip()) > 0):
                    
                    prompt = transcribed_text
                    is_voice_input = True
                    st.success(f"🎤 Voice query transcribed: '{prompt}'")
                else:
                    st.error("😕 Sorry, I couldn't understand the audio. Please try again or type your question.")
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                logger.error(f"Audio processing error: {e}")

# 2. Process text input
if text_prompt := st.chat_input("Ask a question about health topics..."):
    prompt = text_prompt
    is_voice_input = False

# --- Main Logic: If there is a prompt, process it ---
if prompt and vector_store:
    # Add user message to history and display it
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "is_voice": is_voice_input
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from the LLM
                result = get_response(prompt, vector_store)
                answer = result.get("answer", "Sorry, I encountered an error processing your request.")
                source = result.get("source", "N/A")
                lang = result.get("lang", "en")
                detected_lang = result.get("detected_lang", "en")
                
                # Display the answer
                st.markdown(answer)
                st.caption(f"📚 Source: {source} | Language: {detected_lang}")

                # Generate voice response ONLY for voice input
                audio_file_path = None
                if is_voice_input:
                    try:
                        with st.spinner("🔊 Generating audio response..."):
                            audio_file_path = text_to_speech(answer, lang=lang)
                            
                            if os.path.exists(audio_file_path):
                                st.audio(audio_file_path, format="audio/mp3")
                                
                                # Store audio file path for persistence
                                audio_key = f"audio_{len(st.session_state.messages)}"
                                st.session_state.audio_responses[audio_key] = audio_file_path
                            else:
                                st.warning("Audio generation completed but file not found")
                                
                    except Exception as e:
                        st.warning(f"🔊 Audio playback not available: {str(e)}")
                        logger.error(f"TTS failed: {e}")

                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "source": source,
                    "has_audio": is_voice_input and audio_file_path is not None
                })
                
            except Exception as e:
                error_msg = "I apologize, but I encountered an error while processing your question. Please try again."
                st.error(f"❌ {error_msg}")
                logger.error(f"Error in main processing loop: {e}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg, 
                    "source": "error",
                    "has_audio": False
                })
    
    # Rerun to update the chat display
    st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🏥 Lighthouse HealthConnect - Your AI Health Assistant<br>
        Supporting English, Yoruba, Igbo, Hausa, and Nigerian Pidgin<br>
        Powered by advanced language models and vector search
    </div>
    """, 
    unsafe_allow_html=True
)