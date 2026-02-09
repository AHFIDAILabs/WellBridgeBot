from fileinput import filename
import json
import os
import requests
import hashlib
import logging
import aiohttp
import re
from flask import current_app

logger = logging.getLogger(__name__)


def format_text_for_whatsapp(text: str) -> str:
    """
    Convert markdown/rich text formatting to WhatsApp-compatible formatting.
    
    WhatsApp supports:
    - *bold* for bold
    - _italic_ for italic
    - ~strikethrough~ for strikethrough
    - ```code``` for monospace
    
    Args:
        text: Text with markdown formatting
        
    Returns:
        Text formatted for WhatsApp
    """
    if not text:
        return text
    
    # Convert markdown bold (**text** or __text__) to WhatsApp bold (*text*)
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.+?)__', r'*\1*', text)
    
    # Remove markdown headers (###, ##, #) - WhatsApp doesn't support them
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Convert markdown lists to simple bullet points
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
    
    # Remove markdown links [text](url) - show as "text (url)"
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 (\2)', text)
    
    # Clean up excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


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


def send_typing_indicator(recipient_phone_number, message_id):
    """
    Send typing indicator to WhatsApp recipient and mark message as read.
    Shows "typing..." animation for up to 25 seconds or until you send a response.
    
    Args:
        recipient_phone_number: The recipient's phone number
        message_id: The WhatsApp message ID from the incoming message
        
    Returns:
        Response from WhatsApp API
    """
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    data = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"⌨️  Typing indicator sent to {recipient_phone_number}")
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to send typing indicator: {e}")
        return None


def send_recording_indicator(recipient_phone_number, message_id):
    """
    Send recording indicator to WhatsApp recipient and mark message as read.
    Shows microphone/recording animation for up to 25 seconds or until you send a response.
    
    Args:
        recipient_phone_number: The recipient's phone number
        message_id: The WhatsApp message ID from the incoming message
        
    Returns:
        Response from WhatsApp API
    """
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    data = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "audio"
        }
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"🎤 Recording indicator sent to {recipient_phone_number}")
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to send recording indicator: {e}")
        return None


def send_text_message(text, recipient_phone_number):
    """
    Send a text message to a WhatsApp recipient.
    Automatically formats text for WhatsApp compatibility.
    
    Args:
        text: The text message to send (can include markdown)
        recipient_phone_number: The recipient's phone number
        
    Returns:
        Response from WhatsApp API
    """
    # Format text for WhatsApp
    formatted_text = format_text_for_whatsapp(text)
    
    url = f"https://graph.facebook.com/{os.getenv('VERSION')}/{os.getenv('PHONE_NUMBER_ID')}/messages"

    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone_number,
        "type": "text",
        "text": {
            "body": formatted_text
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


def send_language_selection_menu(recipient_phone_number, current_lang=None):
    """
    Send language selection menu with interactive buttons.
    For first-time users (current_lang=None): Shows Hausa, Igbo, Yoruba (English is default if no selection)
    For language switching: Excludes current language to fit WhatsApp's 3-button limit.
    
    Args:
        recipient_phone_number: The recipient's phone number
        current_lang: Current language code to exclude from options (None for first-time users)
        
    Returns:
        Response from WhatsApp API
    """
    # All available languages
    all_languages = [
        {"id": "lang_en", "title": "English", "code": "en"},
        {"id": "lang_ha", "title": "Hausa", "code": "ha"},
        {"id": "lang_ig", "title": "Igbo", "code": "ig"},
        {"id": "lang_yo", "title": "Yoruba", "code": "yo"},
    ]
    
    # For first-time users, show only Nigerian languages (Hausa, Igbo, Yoruba)
    if current_lang is None:
        available_languages = [lang for lang in all_languages if lang["code"] in ["ha", "ig", "yo"]]
        body_text = (
            "Hey! 👋 I'm *WellBridge TB Health Bot* 🏥\n\n"
            "I'm here to help answer your questions about Tuberculosis (TB) - "
            "symptoms, prevention, treatment, and more!\n\n"
            "🌍 I'm set to *English* by default.\n\n"
            "💬 You can start asking questions right away, or select a different language:\n\n"
        )
    else:
        # For language switching, exclude current language
        available_languages = [lang for lang in all_languages if lang["code"] != current_lang]
        available_languages = available_languages[:3]  # Take only first 3 for WhatsApp limit
        body_text = (
            "Hey! 👋 I'm *WellBridge TB Health Bot* 🏥\n\n"
            "I'm here to help answer your questions about Tuberculosis (TB) - "
            "symptoms, prevention, treatment, and more!\n\n"
            "🌍 Select your preferred language:\n\n"
        )
    
    # Add language options to text
    for lang in available_languages:
        body_text += f"🔹 {lang['title']}\n"
    
    # Add instruction for changing language later (only for first-time users)
    if current_lang is None:
        body_text += "\n💡 *To change language later:* Send \"change language\""
    
    buttons = [{"id": lang["id"], "title": lang["title"]} for lang in available_languages]
    
    return send_interactive_buttons(recipient_phone_number, body_text, buttons)



def send_language_options_after_answer(recipient_phone_number):
    """
    Send a brief language options menu after answering a first-time user's question.
    This is less verbose than the full welcome menu.
    
    Args:
        recipient_phone_number: The recipient's phone number
        
    Returns:
        Response from WhatsApp API
    """
    body_text = (
        "🌍 *Language Options*\n\n"
        "I'm currently set to *English*. You can continue in English or switch to:\n\n"
        "🔹 Hausa\n"
        "🔹 Igbo\n"
        "🔹 Yoruba"
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
            "✅ *Madalla! An saita Hausa a matsayin yarenku!*\n\n"
            "Ina shirye in taimaka! Ku tambaye ni komai game da cutar TB kamar:\n"
            "• Menene cutar tarin fuka (TB)?\n"
            "• Ta yaya ake yaduwar cutar TB?\n"
            "• Menene alamomin cutar TB?\n"
            "• Ta yaya zan kare kaina daga cutar TB?\n"
            "• Wane irin magani ake bayarwa?\n\n"
            "💬 Rubuta tambayanku zan bayar da amsa mai amfani!\n\n"
            "💡 *Don canza yare:* Aika \"canza yare\""
        ),
        "yo": (
            "✅ *Ó dára! Yorùbá ti wà gẹ́gẹ́ bíi èdè rẹ!*\n\n"
            "Mo ti ṣetan láti ràn ọ́ lọ́wọ́! O lè béèrè lọ́wọ́ mi nípa TB bíi:\n"
            "• Kí ni TB (àrùn ẹ̀dọ̀fóró)?\n"
            "• Báwo ni TB ṣe ń ràn kálẹ̀?\n"
            "• Kí ni àwọn àmì àrùn TB?\n"
            "• Báwo ni èmi ṣe lè ṣe ìdáàbòbò ara mi lọ́wọ́ TB?\n"
            "• Irú ìtọ́jú wo ni ó wà?\n\n"
            "💬 Kọ ìbéèrè rẹ, èmi yóò sì dáhùn pẹ̀lú àlàyé!\n\n"
            "💡 *Láti yí èdè padà:* Fi ránṣẹ́ \"yi ipada ede\""
        ),
        "ig": (
            "✅ *Ọ dị mma! A tọrọ Ìgbò dịka asụsụ gị!*\n\n"
            "Adị m njikere inyere gị aka! Ị nwere ike ịjụ m ihe ọ bụla gbasara TB dị ka:\n"
            "• Kedu ihe bụ TB (ọrịa nku)?\n"
            "• Kedu ka TB si agbasa?\n"
            "• Kedu ihe bụ ihe ngosi nke TB?\n"
            "• Kedu ka m ga-esi gbochie TB?\n"
            "• Kedu ụdị ọgwụgwọ dị?\n\n"
            "💬 Dee ajụjụ gị, m ga-aza ya na nkọwa!\n\n"
            "💡 *Iji gbanwee asụsụ:* Zipu \"gbanwee asụsụ\""
        ),
        "en": (
            "✅ *Great! English has been set as your language!*\n\n"
            "I'm ready to help! You can ask me anything about TB like:\n"
            "• What is tuberculosis?\n"
            "• How is TB transmitted?\n"
            "• What are the symptoms of TB?\n"
            "• How can I prevent TB?\n"
            "• What treatments are available?\n\n"
            "💬 Just type your question and I'll respond with helpful information!\n\n"
            "💡 *To change language later:* Send \"change language\""
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


def send_followup_questions(recipient_phone_number, lang="en"):
    """
    Send follow-up question buttons after answering a user query.
    
    Args:
        recipient_phone_number: The recipient's phone number
        lang: Language code (en, ha, yo, ig)
        
    Returns:
        Response from WhatsApp API
    """
    followup_messages = {
        "en": {
            "text": "💡 *Want to learn more?* Here are some related questions:",
            "buttons": [
                {"id": "q_transmission", "title": "How TB spreads?"},
                {"id": "q_symptoms", "title": "TB symptoms?"},
                {"id": "q_prevention", "title": "Prevent TB?"},
            ]
        },
        "ha": {
            "text": "💡 *Kuna son ƙarin bayani?* Ga wasu tambayoyi masu alaƙa:",
            "buttons": [
                {"id": "q_transmission_ha", "title": "Yadda TB ke yaduwa?"},
                {"id": "q_symptoms_ha", "title": "Alamomin TB?"},
                {"id": "q_prevention_ha", "title": "Kare TB?"},
            ]
        },
        "yo": {
            "text": "💡 *Ṣe o fẹ́ kọ́ síi?* Èyí ni àwọn ìbéèrè tó jọmọ́:",
            "buttons": [
                {"id": "q_transmission_yo", "title": "Bí TB ṣe ń ràn?"},
                {"id": "q_symptoms_yo", "title": "Àmì TB?"},
                {"id": "q_prevention_yo", "title": "Dáàbò TB?"},
            ]
        },
        "ig": {
            "text": "💡 *Ị chọrọ ịmụta ọzọ?* Nke a bụ ajụjụ ndị metụtara ya:",
            "buttons": [
                {"id": "q_transmission_ig", "title": "Ka TB si agbasa?"},
                {"id": "q_symptoms_ig", "title": "Ihe ngosi TB?"},
                {"id": "q_prevention_ig", "title": "Gbochie TB?"},
            ]
        }
    }
    
    # Get message for language, default to English
    message_data = followup_messages.get(lang, followup_messages["en"])
    
    return send_interactive_buttons(
        recipient_phone_number,
        message_data["text"],
        message_data["buttons"]
    )



