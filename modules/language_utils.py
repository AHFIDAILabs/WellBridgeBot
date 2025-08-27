# modules/language_utils.py: Enhanced language detection utilities
import re
from langdetect import detect, LangDetectException

# Yoruba language patterns and keywords
YORUBA_PATTERNS = {
    'diacritics': r'[àáèéìíòóùúẹọṣ]',
    'common_words': [
        'ṣé', 'bí', 'ní', 'sí', 'tí', 'kí', 'wá', 'ló', 'àti', 'pé',
        'àrùn', 'ìwòsàn', 'ààìsàn', 'ọ̀kan', 'méjì', 'mẹ́ta', 'mẹ́rin',
        'àpótí', 'ilé', 'ọmọ', 'baba', 'mama', 'ẹni', 'àwọn', 'gbogbo',
        'dára', 'burú', 'púpọ̀', 'díẹ̀', 'jẹ', 'mu', 'lọ', 'wá', 'rí'
    ],
    'question_words': ['ṣé', 'báwo', 'ibo', 'nígbà', 'kí', 'ta'],
    'health_terms': ['àrùn', 'ìwòsàn', 'ààìsàn', 'TB', 'ọgbẹ́ni', 'ẹ̀jẹ̀']
}

# Igbo language patterns
IGBO_PATTERNS = {
    'diacritics': r'[àáèéìíòóùúụọ]',
    'common_words': [
        'nke', 'na', 'ọ', 'bụ', 'ga', 'nwere', 'ka', 'maka', 'site',
        'mgbe', 'ebe', 'gịnị', 'onye', 'ndị', 'oge', 'ụbọchị',
        'ọrịa', 'ahụike', 'ọgwụ', 'dọkịta', 'ụlọ', 'mmadụ'
    ],
    'question_words': ['gịnị', 'ebee', 'mgbe', 'kedụ', 'ole'],
    'health_terms': ['ọrịa', 'ahụike', 'ọgwụ', 'TB', 'ụkwara']
}

# Hausa language patterns
HAUSA_PATTERNS = {
    'diacritics': r'[àáèéìíòóùúũñ]',
    'common_words': [
        'da', 'ba', 'ya', 'ta', 'na', 'ka', 'ki', 'mu', 'ku', 'su',
        'wannan', 'waccan', 'lokaci', 'rana', 'dare', 'mutane',
        'cuta', 'lafiya', 'magani', 'likita', 'gida', 'yara'
    ],
    'question_words': ['me', 'ina', 'yaushe', 'yaya', 'wace'],
    'health_terms': ['cuta', 'lafiya', 'magani', 'TB', 'tari']
}

def detect_yoruba(text):
    """Specifically detect Yoruba language with stricter criteria"""
    text_lower = text.lower()
    
    # Strong indicator: Check for Yoruba diacritics
    diacritic_count = len(re.findall(YORUBA_PATTERNS['diacritics'], text))
    if diacritic_count >= 3:  # Need at least 3 diacritics to be confident
        return True
    
    # Check for common Yoruba words
    yoruba_word_count = 0
    words = text_lower.split()
    for word in words:
        if word in YORUBA_PATTERNS['common_words']:
            yoruba_word_count += 1
    
    # Need at least 30% of words to be Yoruba (increased threshold)
    if len(words) > 0 and (yoruba_word_count / len(words)) > 0.3:
        return True
    
    return False

def detect_igbo(text):
    """Specifically detect Igbo language with stricter criteria"""
    text_lower = text.lower()
    
    # Check for Igbo-specific patterns
    igbo_word_count = 0
    words = text_lower.split()
    for word in words:
        if word in IGBO_PATTERNS['common_words']:
            igbo_word_count += 1
    
    # Need at least 30% of words to be Igbo
    if len(words) > 0 and (igbo_word_count / len(words)) > 0.3:
        return True
    
    return False

def detect_hausa(text):
    """Specifically detect Hausa language with stricter criteria"""
    text_lower = text.lower()
    
    # Check for common Hausa words
    hausa_word_count = 0
    words = text_lower.split()
    for word in words:
        if word in HAUSA_PATTERNS['common_words']:
            hausa_word_count += 1
    
    # Need at least 30% of words to be Hausa
    if len(words) > 0 and (hausa_word_count / len(words)) > 0.3:
        return True
    
    return False

def detect_language(text):
    """Enhanced language detection with Nigerian language support"""
    if not text or not text.strip():
        return "en"
    
    text = text.strip()
    
    # First, try langdetect to get a baseline
    try:
        detected_base = detect(text)
    except LangDetectException:
        detected_base = "en"
    
    # If langdetect confidently detects English, check if it's actually Pidgin
    if detected_base == "en":
        # Check if it's actually Pidgin English
        if is_pidgin(text):
            return "pidgin"
        # Check if there are strong indicators of Nigerian languages
        if detect_yoruba(text):
            return "yo"
        elif detect_igbo(text):
            return "ig"
        elif detect_hausa(text):
            return "ha"
        # Otherwise, it's standard English
        return "en"
    
    # If langdetect detects Yoruba, Igbo, or Hausa, verify with our patterns
    elif detected_base == "yo":
        if detect_yoruba(text):
            return "yo"
        return "en"  # Default to English if not confirmed
    
    elif detected_base == "ig":
        if detect_igbo(text):
            return "ig"
        return "en"
    
    elif detected_base == "ha":
        if detect_hausa(text):
            return "ha"
        return "en"
    
    # For any other detected language, check our Nigerian patterns
    else:
        if detect_yoruba(text):
            return "yo"
        elif detect_igbo(text):
            return "ig"
        elif detect_hausa(text):
            return "ha"
        elif is_pidgin(text):
            return "pidgin"
    
    # Default to English
    return "en"

def is_pidgin(text):
    """Enhanced Pidgin detection with stricter criteria"""
    pidgin_keywords = [
        'dey', 'no go', 'una', 'make we', 'abi', 'wey', 'na why',
        'e be like', 'wetin', 'comot', 'waka', 'how far', 'no vex',
        'i dey', 'you dey', 'dem dey', 'na so', 'shey', 'sef',
        'wahala', 'pikin', 'oya', 'chop', 'yarn', 'gbagaun'
    ]
    
    text_lower = text.lower()
    pidgin_count = 0
    
    for keyword in pidgin_keywords:
        if keyword in text_lower:
            pidgin_count += 1
    
    # Need stronger evidence for Pidgin (at least 3 keywords or specific phrases)
    return pidgin_count >= 3 or any(phrase in text_lower for phrase in ['i wan', 'make i', 'no be', 'e dey'])

def get_language_name(lang_code):
    """Get human-readable language name"""
    lang_names = {
        'yo': 'Yoruba',
        'ig': 'Igbo',
        'ha': 'Hausa',
        'en': 'English',
        'pidgin': 'Nigerian Pidgin'
    }
    return lang_names.get(lang_code, 'English')  # Default to English name