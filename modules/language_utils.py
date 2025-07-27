# modules/language_utils.py: 
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "en"  # Fallback if detection fails

def is_pidgin(text):
    pidgin_keywords = [
        "dey", "no go", "una", "make we", "abi", "wey", "na why",
        "e be like", "wetin", "comot", "waka", "how far", "no vex"
    ]
    text = text.lower()
    return any(keyword in text for keyword in pidgin_keywords)