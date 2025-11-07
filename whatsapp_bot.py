from flask import Flask, request, jsonify
import os
import requests
import logging
import tempfile
import uuid
from dotenv import load_dotenv
from modules.audio_handler import AudioHandler
from modules.llm_handler import LLMHandler
# from modules.language_utils import detect_language, is_pidgin
from modules.language_selection import LanguageSelectionHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Initialize handlers
audio_handler = AudioHandler()
llm_handler = LLMHandler()
language_selection_handler = LanguageSelectionHandler()

# WhatsApp API Configuration
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# Validate required environment variables
if not all([VERIFY_TOKEN, WHATSAPP_TOKEN, PHONE_NUMBER_ID]):
    raise ValueError("Missing required environment variables. Please check .env file")

WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"

def get_media_url(media_id: str) -> str:
    """Get media URL from WhatsApp"""
    url = f"https://graph.facebook.com/v17.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("url")
    except Exception as e:
        logger.error(f"Error getting media URL: {str(e)}")
        return None

def download_media(url: str) -> str:
    """Download media from WhatsApp"""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Save to temporary file
        temp_path = os.path.join(tempfile.gettempdir(), f"whatsapp_audio_{uuid.uuid4()}.ogg")
        with open(temp_path, "wb") as f:
            f.write(response.content)
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading media: {str(e)}")
        return None

def send_whatsapp_message(phone_number: str, message: str, is_audio: bool = False):
    """Send text message to WhatsApp"""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message}
    }
    
    try:
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        return None

def send_whatsapp_audio(phone_number: str, audio_path: str):
    """Send audio message to WhatsApp"""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    }
    
    try:
        # First upload the audio file
        files = {
            'file': ('audio.mp3', open(audio_path, 'rb'), 'audio/mp3')
        }
        upload_response = requests.post(
            f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files
        )
        upload_response.raise_for_status()
        media_id = upload_response.json()['id']
        
        # Then send the audio message
        data = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "audio",
            "audio": {"id": media_id}
        }
        
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error sending WhatsApp audio: {str(e)}")
        return None

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == "GET":
        # Handle webhook verification
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        
        if mode and token:
            if mode == "subscribe" and token == VERIFY_TOKEN:
                logger.info("Webhook verified!")
                return challenge
            else:
                return "Forbidden", 403
                
    elif request.method == "POST":
        data = request.get_json()
        
        try:
            if data["object"]:
                if data["entry"][0]["changes"][0]["value"].get("messages"):
                    phone_number = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
                    message_type = data["entry"][0]["changes"][0]["value"]["messages"][0]["type"]
                    
                    if message_type == "audio":
                        # Handle voice message
                        audio_id = data["entry"][0]["changes"][0]["value"]["messages"][0]["audio"]["id"]
                        audio_url = get_media_url(audio_id)
                        
                        # Download and process audio
                        audio_path = download_media(audio_url)
                        if audio_path:
                            text = audio_handler.speech_to_text(audio_path)
                            os.remove(audio_path)  # Clean up
                    else:
                        # Handle text message
                        text = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

                        # Language selection handling
                        response, selected_language, should_process = language_selection_handler.handle_message(phone_number, text)
                        
                        if response:
                            # Handle language selection response
                            send_whatsapp_message(phone_number, response)
                            
                            # Generate audio for language selection response if needed
                            audio_path = audio_handler.text_to_speech(response, selected_language or 'en')
                            if audio_path:
                                send_whatsapp_audio(phone_number, audio_path)
                                os.remove(audio_path)  # Clean up
                            
                            if not should_process:
                                return "OK", 200

                        # Process message with LLM if we should continue
                        if should_process:
                            # Use selected language for LLM processing
                            llm_response = llm_handler.process_message(text, selected_language)
                            
                            if llm_response:
                                # Send text response
                                send_whatsapp_message(phone_number, llm_response)
                                
                                # Generate and send audio response
                                audio_path = audio_handler.text_to_speech(llm_response, selected_language)
                                if audio_path:
                                    send_whatsapp_audio(phone_number, audio_path)
                                    os.remove(audio_path)  # Clean up

                return "OK", 200

        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return "OK", 200  # Always return 200 to WhatsApp
            
    return "OK", 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "audio_handler": bool(audio_handler),
        "llm_handler": bool(llm_handler),
        "whatsapp_api": bool(WHATSAPP_TOKEN)
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)