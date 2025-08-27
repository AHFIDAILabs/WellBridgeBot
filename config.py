# config.py: Enhanced configuration for multilingual Nigerian health chatbot
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Pinecone Configuration
PINECONE_INDEX_NAME = "lighthouse-healthconnect-multilingual"
TEXT_KEY = "text"

# Embedding Model (multilingual support)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM Configuration
LLM_MODEL = "openai/gpt-4o-mini"  # Better for multilingual tasks than free models

# Language Configuration
SUPPORTED_LANGUAGES = {
    "yo": {
        "name": "Yoruba",
        "native_name": "Yorùbá",
        "has_tts": True,
        "tts_lang": "yo",
        "fallback_tts": "en"
    },
    "ig": {
        "name": "Igbo", 
        "native_name": "Igbo",
        "has_tts": False,
        "tts_lang": "en",
        "fallback_tts": "en"
    },
    "ha": {
        "name": "Hausa",
        "native_name": "Hausa", 
        "has_tts": True,
        "tts_lang": "ha",
        "fallback_tts": "en"
    },
    "pidgin": {
        "name": "Nigerian Pidgin",
        "native_name": "Naija Pidgin",
        "has_tts": False,
        "tts_lang": "en",
        "fallback_tts": "en"
    },
    "en": {
        "name": "English",
        "native_name": "English",
        "has_tts": True,
        "tts_lang": "en", 
        "fallback_tts": "en"
    }
}

# Audio Configuration
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
AUDIO_TEMP_DIR = os.getenv("AUDIO_TEMP_DIR", None)  # None uses system temp
AUDIO_CLEANUP_DELAY = 300  # seconds before cleanup

# Retrieval Configuration
RETRIEVAL_K = 5  # Number of documents to retrieve
RETRIEVAL_SEARCH_TYPE = "similarity"  # or "mmr"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Response Configuration
MAX_RESPONSE_LENGTH = 600  # tokens
TEMPERATURE = 0.2
RESPONSE_TIMEOUT = 35  # seconds

 # Threshold for KB result quality before falling back to internet search
MIN_SCORE = 1.5 

# Health/Medical Configuration
HEALTH_KEYWORDS = [
    # English
    "tuberculosis", "TB", "disease", "infection", "treatment", "prevention", 
    "symptoms", "cure", "medicine", "doctor", "hospital",
    
    # Yoruba
    "àrùn", "ìwòsàn", "àìsàn", "ọgbẹ́ni", "ẹ̀jẹ̀", "ara", "ilera",
    
    # Igbo
    "ọrịa", "ahụike", "ọgwụ", "dọkịta", "ụlọ", "ara", "nsogbu",
    
    # Hausa
    "cuta", "lafiya", "magani", "likita", "jiki", "matsala",
    
    # Pidgin
    "sickness", "sick", "well", "medicine", "doctor", "hospital", "body"
]

# Validation
def validate_config():
    """Validate configuration settings."""
    missing_keys = []
    
    if not OPENROUTER_API_KEY:
        missing_keys.append("OPENROUTER_API_KEY")
    
    if not PINECONE_API_KEY:
        missing_keys.append("PINECONE_API_KEY")
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return True

def get_language_config(lang_code):
    """Get configuration for specific language."""
    return SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])

def is_supported_language(lang_code):
    """Check if language is supported."""
    return lang_code in SUPPORTED_LANGUAGES

def get_tts_config(lang_code):
    """Get TTS configuration for language."""
    config = get_language_config(lang_code)
    return {
        "lang": config["tts_lang"],
        "fallback": config["fallback_tts"],
        "has_native": config["has_tts"]
    }

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "development":
    LLM_MODEL = "openai/gpt-oss-20b:free"  # Use free model for development
    RETRIEVAL_K = 3  # Fewer retrievals for faster testing
    MAX_RESPONSE_LENGTH = 400
    
elif os.getenv("ENVIRONMENT") == "production":
    LLM_MODEL = "openai/gpt-4o-mini"  # Better model for production
    RETRIEVAL_K = 5
    MAX_RESPONSE_LENGTH = 600