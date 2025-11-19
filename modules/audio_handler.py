# modules/audio_handler.py: Multilingual audio handler for Nigerian languages
"""
REFACTORED: This module now provides backward-compatible wrapper functions
around the new class-based audio processing system.

New architecture:
- AudioTranscriber: Handles speech-to-text (audio_transcriber.py)
- AudioSynthesizer: Handles text-to-speech (audio_synthesizer.py)

These wrappers maintain the original function signatures for compatibility
with existing code (app.py, app-st.py, etc.)
"""

import os
import logging
from typing import Tuple

from modules.audio_transcriber import get_transcriber
from modules.audio_synthesizer import get_synthesizer
from modules.language_utils import get_language_name

logger = logging.getLogger(__name__)


def speech_to_text(audio_file_path: str, selected_lang: str) -> Tuple[str, str]:
    """
    Backward-compatible wrapper for speech-to-text transcription.
    
    Uses the new class-based AudioTranscriber system with intelligent fallback:
    Priority: N-ATLAS (Nigerian languages) -> Whisper -> Google Speech Recognition
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        selected_lang: Language code ("auto" for auto-detection, or specific code)
    
    Returns:
        Tuple of (transcribed_text, detected_language)
    
    Example:
        text, lang = speech_to_text("audio.wav", "ha")
        text, lang = speech_to_text("audio.wav", "auto")
    """
    try:
        transcriber = get_transcriber()
        text, detected_lang = transcriber.transcribe(audio_file_path, selected_lang)
        
        logger.info(f"Transcription successful: lang={detected_lang}, text='{text[:50]}...'")
        return text, detected_lang
        
    except Exception as e:
        logger.error(f"Speech-to-text failed: {e}")
        # Return error message for backward compatibility
        return f"Transcription failed: {str(e)}", selected_lang


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Backward-compatible wrapper for text-to-speech synthesis.
    
    Uses the new class-based AudioSynthesizer system:
    - MMS-TTS for Yoruba and Hausa (native support)
    - gTTS for other languages (with Nigerian accent where appropriate)
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: "en")
    
    Returns:
        Path to the generated audio file
    
    Example:
        audio_path = text_to_speech("Hello", "en")
        audio_path = text_to_speech("Bawo ni", "yo")
    """
    try:
        synthesizer = get_synthesizer()
        audio_path = synthesizer.synthesize(text, lang)
        
        logger.info(f"TTS successful for {lang}: {audio_path}")
        return audio_path
        
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise Exception(f"TTS conversion failed: {e}")


def cleanup_audio_file(file_path: str) -> None:
    """
    Clean up temporary audio files.
    
    Backward-compatible wrapper for the AudioSynthesizer cleanup method.
    
    Args:
        file_path: Path to the audio file to delete
    """
    try:
        synthesizer = get_synthesizer()
        synthesizer.cleanup_audio_file(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


def get_supported_tts_languages():
    """
    Return list of languages with TTS support.
    
    This function provides backward compatibility.
    """
    return {
        "yo": "Yoruba (Native TTS)",
        "ha": "Hausa (Native TTS)",
        "ig": "Igbo (Nigerian English TTS)",
        "pidgin": "Nigerian Pidgin (Nigerian English TTS)",
        "en": "English (Nigerian accent)",
    }
