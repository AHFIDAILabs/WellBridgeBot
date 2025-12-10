from fileinput import filename
import json
import os
import requests
import hashlib
import logging
import aiohttp
from flask import current_app

logger = logging.getLogger(__name__)


def parse_json(data):
    """
    Parse WhatsApp webhook payload.
    
    Args:
        data: The JSON payload from WhatsApp webhook
        
    Returns:
        Dictionary with message data or None if not a message event
    """
    try:
        entry = data["entry"][0]
        value = entry["changes"][0]["value"]
        
        # Check if this is a message event (not status update, etc.)
        if "messages" not in value:
            logger.info("ℹ️  Webhook event received (non-message event, e.g., status update)")
            print(data)
            return None
        
        msg = value["messages"][0]

        wa_id = msg["from"]
        msg_id = msg["id"]

        # determine message type

        if msg["type"] == "text":
            text = msg["text"]["body"]
            message = {
                "type": "text",
                "wa_id": wa_id,
                "id": msg_id,
                "text": text
            }
            print(message)
            return message

        if msg["type"] == "audio":
            audio = msg["audio"]
            message = {
                "type": "audio",
                "wa_id": wa_id,
                "id": msg_id,
                "audio": audio
            }
            return message
        
        if msg["type"] == "interactive":
            # Handle button response
            interactive = msg["interactive"]
            button_reply = interactive.get("button_reply", {})
            message = {
                "type": "interactive",
                "wa_id": wa_id,
                "id": msg_id,
                "button_id": button_reply.get("id"),
                "button_title": button_reply.get("title")
            }
            return message
        
        # Unsupported message type
        logger.info(f"ℹ️  Unsupported message type: {msg['type']}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error parsing webhook JSON: {e}", exc_info=True)
        return None

def get_media_url(media_id):
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{media_id}"

    data = {

    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",

        "Content-Type": "application/json"
        }
    response = requests.request("GET", url, json=data, headers=headers)

    return response.json()

def download_media(json_object):

    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{json_object['url']}"

    data = {

    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    response = requests.request("GET", url, json=data, headers=headers)

    print(response.json())

    filename = json_object['id'] + ".ogg"
    
    r = requests.get(json_object['url'], stream=True, headers=headers)
    with open(filename, "wb") as f:
        for chunk in r.iter_content(4096):
           f.write(chunk)

    return filename

def upload_media(file_path):
    url = "https://graph.facebook.com/v13.0/me/messages"
    headers = {"Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}"}
    with open(file_path, "rb") as f:
        r = requests.post(url, headers=headers, files={"file": f})
    return r.json()


def upload_media_to_whatsapp(file_path: str) -> dict:
    """
    Upload media file to WhatsApp and get media ID.
    Supports OGG (preferred), MP3, WAV, and other audio formats.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        dict with 'id' of uploaded media or 'error'
    """
    try:
        import os
        
        # Determine MIME type based on file extension
        if file_path.endswith('.ogg'):
            mime_type = 'audio/ogg; codecs=opus'
        elif file_path.endswith('.mp3'):
            mime_type = 'audio/mpeg'
        elif file_path.endswith('.wav'):
            mime_type = 'audio/wav'
        else:
            mime_type = 'audio/ogg; codecs=opus'  # Default to OGG
        
        url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/media"
        
        headers = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}"
        }
        
        # Read file and upload with correct MIME type
        with open(file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(file_path), audio_file, mime_type),
                "messaging_product": (None, "whatsapp"),
                "type": (None, mime_type)
            }
            
            response = requests.post(url, headers=headers, files=files)
            
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ Media uploaded successfully: {result}")
            return result
        else:
            logger.error(f"❌ Media upload failed: {response.status_code} - {response.text}")
            return {"error": response.text}
            
    except Exception as e:
        logger.error(f"❌ Exception during media upload: {e}", exc_info=True)
        return {"error": str(e)}


async def send_message(data):
    """
    Send a message via WhatsApp asynchronously.
    
    Args:
        data: JSON data for the message
    """
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
    }

    async with aiohttp.ClientSession() as session:
        url = 'https://graph.facebook.com' + f"/{ os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"
        try:
            async with session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    print("Status:", response.status)
                    print("Content-type:", response.headers['content-type'])

                    html = await response.text()
                    print("Body:", html)
                else:
                    print(response.status)        
                    print(response)        
        except aiohttp.ClientConnectorError as e:
            print('Connection Error', str(e))


def send_audio_message(audio_object_id, recipient_phone_number):
    """
    Send an audio message to a WhatsApp recipient.
    
    Args:
        audio_object_id: The media ID from WhatsApp after upload
        recipient_phone_number: The recipient's phone number
    """
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    data = {
        "audio": {
            "id": audio_object_id
        },
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone_number,
        "type": "audio"
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=data, headers=headers)

    print(response.json())


def send_text_message(text, recipient_phone_number):
    """
    Send a text message to a WhatsApp recipient.
    
    Args:
        text: The text message to send
        recipient_phone_number: The recipient's phone number
        
    Returns:
        Response from WhatsApp API
    """
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone_number,
        "type": "text",
        "text": {
            "body": text
        }
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=data, headers=headers)

    print(response.json())
    return response.json()


def get_text_message_input(recipient, text):
    """
    Format a text message for WhatsApp API.
    
    Args:
        recipient: The recipient's phone number
        text: The message text
        
    Returns:
        JSON string formatted for WhatsApp API
    """
    return json.dumps({
        "messaging_product": "whatsapp",
        "preview_url": False,
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {
            "body": text
        }
    })


def send_interactive_buttons(recipient_phone_number, body_text, buttons):
    """
    Send an interactive button message to a WhatsApp recipient.
    
    Args:
        recipient_phone_number: The recipient's phone number
        body_text: The message body text
        buttons: List of button dictionaries with 'id' and 'title' keys
                 Example: [{"id": "btn_1", "title": "Button 1"}]
        
    Returns:
        Response from WhatsApp API
    """
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    # Format buttons for WhatsApp API
    action_buttons = [
        {
            "type": "reply",
            "reply": {
                "id": btn["id"],
                "title": btn["title"]
            }
        }
        for btn in buttons
    ]

    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": body_text
            },
            "action": {
                "buttons": action_buttons
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=data, headers=headers)

    logger.info(f"Interactive button response: {response.json()}")
    return response.json()


def send_language_selection_menu(recipient_phone_number):
    """
    Send language selection menu with interactive buttons.
    
    Args:
        recipient_phone_number: The recipient's phone number
        
    Returns:
        Response from WhatsApp API
    """
    body_text = (
        "Welcome to WellBridge TB Health Bot! 🏥\n\n"
        "Please select your preferred language:\n"
        "🔹 Hausa - Hausa\n"
        "🔹 Igbo - Ìgbò\n"
        "🔹 Yoruba - Yorùbá\n"
    )
    
    buttons = [
        {"id": "lang_ha", "title": "Hausa"},
        {"id": "lang_ig", "title": "Igbo"},
        {"id": "lang_yo", "title": "Yoruba"},
    ]
    
    return send_interactive_buttons(recipient_phone_number, body_text, buttons)


def send_language_switch_confirmation(recipient_phone_number, language_name):
    """
    Send confirmation message after language switch with instructions on how to change language later.
    
    Args:
        recipient_phone_number: The recipient's phone number
        language_name: Name of the selected language
        
    Returns:
        Response from WhatsApp API
    """
    confirmations = {
        "ha": (
            "✅ *An saita Hausa a matsayin yarenku!*\n\n"
            "Yanzu zaku iya yin tambayoyi game da cutar TB a Hausa.\n\n"
            "💡 *Don canza yare a lokacin zance:*\n"
            "• Aika: \"canza yare\"\n"
            "• Ko: \"sauyar da yare\"\n"
            "• Ko kawai: \"yare\""
        ),
        "yo": (
            "✅ *Yorùbá ti wà gẹ́gẹ́ bíi èdè rẹ!*\n\n"
            "Ní báyìí o lè béèrè nípa TB ní Yorùbá.\n\n"
            "💡 *Láti yí èdè padà láàrin ìbánisọ̀rọ̀:*\n"
            "• Fi ránṣẹ́: \"yi ipada ede\"\n"
            "• Tàbí: \"paarọ ede\"\n"
            "• Tàbí nìkan: \"ede\""
        ),
        "ig": (
            "✅ *A tọrọ Ìgbò dịka asụsụ gị!*\n\n"
            "Ugbu a ị nwere ike ịjụ ajụjụ gbasara TB n'Ìgbò.\n\n"
            "💡 *Iji gbanwee asụsụ n'etiti mkparịta ụka:*\n"
            "• Zipu: \"gbanwee asụsụ\"\n"
            "• Ma ọ bụ: \"họrọ asụsụ\"\n"
            "• Ma ọ bụ naanị: \"asụsụ\""
        ),
        "en": (
            "✅ *English has been set as your language!*\n\n"
            "You can now ask questions about TB in English.\n\n"
            "💡 *To change language mid-conversation:*\n"
            "• Send: \"change language\"\n"
            "• Or: \"switch language\"\n"
            "• Or simply: \"language\""
        )
    }
    
    # Extract language code from language_name if it's in format "lang_xx"
    lang_code = language_name.replace("lang_", "") if "lang_" in language_name else language_name
    
    text = confirmations.get(lang_code, confirmations["en"])
    
    return send_text_message(text, recipient_phone_number)


def is_language_switch_request(text: str) -> bool:
    """
    Check if user message is requesting to change language.
    Supports keywords in English, Hausa, Yoruba, and Igbo.
    
    Args:
        text: The user's message text
        
    Returns:
        True if message is a language switch request, False otherwise
    """
    if not text:
        return False
    
    # Normalize text for comparison
    text_lower = text.lower().strip()
    
    # Language switch keywords in multiple languages
    switch_keywords = [
        # English
        "change language",
        "switch language",
        "select language",
        "choose language",
        "language settings",
        "change lang",
        "switch lang",
        
        # Hausa
        "canza yare",
        "sauyar da yare",
        "zaɓi yare",
        "sauya yare",
        
        # Yoruba
        "yi ipada ede",
        "paarọ ede",
        "yan ede",
        "yipada ede",
        
        # Igbo
        "gbanwee asụsụ",
        "họrọ asụsụ",
        "gbanwe asusu",
        
        # Short forms
        "language",
        "lang",
        "yare",
        "ede",
        "asụsụ",
        "asusu"
    ]
    
    # Check if any keyword matches
    for keyword in switch_keywords:
        if keyword in text_lower:
            return True
    
    return False


