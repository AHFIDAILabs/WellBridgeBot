# modules/config_manager.py: Enhanced configuration management with validation
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API credentials and endpoints"""
    openrouter_api_key: str
    pinecone_api_key: str
    pinecone_region: str
    pinecone_cloud: str
    huggingface_token: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        """Load API config from environment variables"""
        load_dotenv()
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            huggingface_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate all required keys are present"""
        errors = []
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is missing")
        if not self.pinecone_api_key:
            errors.append("PINECONE_API_KEY is missing")
        
        return len(errors) == 0, errors


@dataclass
class LanguageConfig:
    """Configuration for a specific language"""
    name: str
    native_name: str
    has_tts: bool
    tts_lang: str
    fallback_tts: str
    has_natlas: bool = False
    tts_tld: str = "com"


@dataclass
class ModelConfig:
    """Model configuration for embeddings and LLM"""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "openai/gpt-4o-mini"
    whisper_model_size: str = "base"
    embedding_dimension: int = 384
    

@dataclass
class RetrievalConfig:
    """Vector store retrieval configuration"""
    retrieval_k: int = 5
    retrieval_search_type: str = "similarity"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_score: float = 1.5


@dataclass
class ResponseConfig:
    """LLM response configuration"""
    max_response_length: int = 600
    temperature: float = 0.2
    response_timeout: int = 35


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    audio_temp_dir: Optional[str] = None
    audio_cleanup_delay: int = 300
    whisper_model_size: str = "base"


@dataclass
class AppConfig:
    """Main application configuration"""
    api: APIConfig
    languages: Dict[str, LanguageConfig]
    models: ModelConfig
    retrieval: RetrievalConfig
    response: ResponseConfig
    audio: AudioConfig
    pinecone_index_name: str = "lighthouse-healthconnect-multilingual"
    text_key: str = "text"
    
    @classmethod
    def load(cls):
        """Load complete application configuration"""
        api_config = APIConfig.from_env()
        
        # Define language configurations
        languages = {
            "yo": LanguageConfig(
                name="Yoruba",
                native_name="Yorùbá",
                has_tts=True,
                tts_lang="yo",
                fallback_tts="en",
                has_natlas=True,
                tts_tld="com"
            ),
            "ig": LanguageConfig(
                name="Igbo",
                native_name="Igbo",
                has_tts=False,
                tts_lang="en",
                fallback_tts="en",
                has_natlas=True,
                tts_tld="com.ng"
            ),
            "ha": LanguageConfig(
                name="Hausa",
                native_name="Hausa",
                has_tts=True,
                tts_lang="ha",
                fallback_tts="en",
                has_natlas=True,
                tts_tld="com"
            ),
            "pidgin": LanguageConfig(
                name="Nigerian Pidgin",
                native_name="Naija Pidgin",
                has_tts=False,
                tts_lang="en",
                fallback_tts="en",
                has_natlas=False,
                tts_tld="com.ng"
            ),
            "en": LanguageConfig(
                name="English",
                native_name="English",
                has_tts=True,
                tts_lang="en",
                fallback_tts="en",
                has_natlas=False,
                tts_tld="com.ng"
            )
        }
        
        return cls(
            api=api_config,
            languages=languages,
            models=ModelConfig(),
            retrieval=RetrievalConfig(),
            response=ResponseConfig(),
            audio=AudioConfig(
                audio_temp_dir=os.getenv("AUDIO_TEMP_DIR"),
                whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base")
            )
        )
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate entire configuration"""
        is_valid, errors = self.api.validate()
        
        if not is_valid:
            logger.error(f"Configuration validation failed: {errors}")
        else:
            logger.info("Configuration validated successfully")
        
        return is_valid, errors
    
    def get_language_config(self, lang_code: str) -> Optional[LanguageConfig]:
        """Get configuration for a specific language"""
        return self.languages.get(lang_code)
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return list(self.languages.keys())
    
    def get_natlas_languages(self) -> list[str]:
        """Get list of languages with N-ATLAS support"""
        return [code for code, config in self.languages.items() if config.has_natlas]


# Global configuration instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig.load()
        is_valid, errors = _config_instance.validate()
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {errors}")
    return _config_instance


def reload_config():
    """Reload configuration from environment (useful for testing)"""
    global _config_instance
    _config_instance = None
    return get_config()


# Health/Medical Keywords (moved from config.py)
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
