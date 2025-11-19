# modules/audio_synthesizer.py: Text-to-speech synthesis for multilingual output
"""
AudioSynthesizer: Handles text-to-speech generation with Nigerian language support.
Uses MMS-TTS for native support and gTTS for fallback.
"""

import os
import uuid
import tempfile
import logging
from typing import Optional
import torch
import torchaudio
from gtts import gTTS

from modules.exceptions import TextToSpeechError
from modules.config_manager import get_config

logger = logging.getLogger(__name__)


class MMSynthesizer:
    """Meta Multilingual Speech (MMS) TTS for Yoruba and Hausa"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.enabled_langs = set()
        logger.info("MMS Synthesizer initialized")
    
    def can_synthesize(self, lang: str) -> bool:
        """Check if MMS supports this language"""
        return lang in ["yo", "ha"]
    
    def _load_model(self, lang: str):
        """Lazy load MMS model for a specific language"""
        if lang in self.models:
            return
        
        try:
            from transformers import VitsModel, AutoTokenizer
            
            mms_lang_map = {
                "yo": "yor",  # Yoruba
                "ha": "hau"   # Hausa
            }
            
            if lang not in mms_lang_map:
                raise TextToSpeechError(f"MMS not supported for {lang}")
            
            mms_code = mms_lang_map[lang]
            model_name = f"facebook/mms-tts-{mms_code}"
            
            logger.info(f"Loading MMS model for {lang}...")
            self.models[lang] = VitsModel.from_pretrained(model_name)
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self.enabled_langs.add(lang)
            logger.info(f"✓ MMS {lang.upper()} model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load MMS model for {lang}: {e}")
            raise TextToSpeechError(f"MMS model load failed: {e}")
    
    def synthesize(self, text: str, lang: str) -> str:
        """
        Synthesize speech using MMS
        Returns: Path to generated audio file
        """
        if not self.can_synthesize(lang):
            raise TextToSpeechError(f"MMS cannot synthesize {lang}")
        
        try:
            self._load_model(lang)
            
            logger.info(f"MMS synthesizing {lang}...")
            
            # Tokenize input
            inputs = self.tokenizers[lang](text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                output = self.models[lang](**inputs).waveform
            
            # Save to OGG instead of WAV for WhatsApp compatibility
            temp_dir = tempfile.gettempdir()
            wav_path = os.path.join(temp_dir, f"tts_mms_{uuid.uuid4().hex}.wav")
            ogg_path = os.path.join(temp_dir, f"tts_mms_{uuid.uuid4().hex}.ogg")
            
            # Save as WAV first
            torchaudio.save(wav_path, output, sample_rate=16000)
            
            # Convert to OGG Opus for WhatsApp
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                audio.export(
                    ogg_path, 
                    format='ogg', 
                    codec='libopus',
                    parameters=["-strict", "-2"]
                )
                os.remove(wav_path)  # Remove temporary WAV
                file_path = ogg_path
                logger.info(f"MMS synthesis complete (OGG): {file_path}")
            except ImportError:
                # If pydub not available, use WAV
                file_path = wav_path
                logger.warning("pydub not available, using WAV format")
                logger.info(f"MMS synthesis complete (WAV): {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"MMS synthesis failed: {e}")
            raise TextToSpeechError(f"MMS error: {e}")


class GTTSynthesizer:
    """Google TTS synthesizer (fallback)"""
    
    def synthesize(self, text: str, lang: str = "en", tld: str = "com.ng") -> str:
        """
        Synthesize speech using gTTS
        Returns: Path to generated audio file (OGG format)
        """
        try:
            logger.info(f"gTTS synthesizing {lang} (tld={tld})...")
            
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            
            temp_dir = tempfile.gettempdir()
            mp3_path = os.path.join(temp_dir, f"tts_gtts_{uuid.uuid4().hex}.mp3")
            ogg_path = os.path.join(temp_dir, f"tts_gtts_{uuid.uuid4().hex}.ogg")
            
            # Save as MP3 first
            tts.save(mp3_path)
            
            # Convert to OGG Opus for WhatsApp
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(
                    ogg_path, 
                    format='ogg', 
                    codec='libopus',
                    parameters=["-strict", "-2"]
                )
                os.remove(mp3_path)  # Remove temporary MP3
                file_path = ogg_path
                logger.info(f"gTTS synthesis complete (OGG): {file_path}")
            except Exception as conv_error:
                # If conversion fails, use MP3
                file_path = mp3_path
                logger.warning(f"OGG conversion failed, using MP3: {conv_error}")
                logger.info(f"gTTS synthesis complete (MP3): {file_path}")
            
            logger.info(f"gTTS synthesis complete: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise TextToSpeechError(f"gTTS error: {e}")


class AudioSynthesizer:
    """
    Main TTS class with intelligent provider selection.
    Uses MMS for Yoruba/Hausa, gTTS for others.
    """

    def __init__(self):
        self.mms = MMSynthesizer()
        self.gtts = GTTSynthesizer()
        logger.info("AudioSynthesizer initialized")
    
    def synthesize(self, text: str, lang: str = "en") -> str:
        """
        Convert text to speech
        
        Args: 
            lang: Language code
        
        Returns:
            Path to generated audio file
        
        Raises:
            TextToSpeechError: If synthesis fails
        """
        text = text.strip()
        if not text:
            raise TextToSpeechError("Empty text provided for TTS")
        
        logger.info(f"Synthesizing speech for {lang}: '{text[:50]}...'")
        
        config = get_config()
        lang_config = config.get_language_config(lang)
        
        if not lang_config:
            logger.warning(f"Unknown language {lang}, using English")
            lang = "en"
            lang_config = config.get_language_config("en")
        
        # Try MMS for Yoruba and Hausa
        if lang in ["yo", "ha"]:
            try:
                return self.mms.synthesize(text, lang)
            except TextToSpeechError as e:
                logger.warning(f"MMS failed for {lang}, falling back to gTTS: {e}")
        
        # Use gTTS for all other languages
        try:
            tts_lang = lang_config.tts_lang
            tts_tld = lang_config.tts_tld
            return self.gtts.synthesize(text, tts_lang, tts_tld)
        except TextToSpeechError as e:
            # Final fallback to English
            logger.error(f"gTTS failed for {lang}, trying English fallback")
            try:
                return self.gtts.synthesize(text, "en", "com.ng")
            except Exception as e2:
                raise TextToSpeechError(f"All TTS methods failed: {e2}")
    
    def cleanup_audio_file(self, file_path: str):
        """Clean up temporary audio file"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


# Singleton instance
_synthesizer_instance: Optional[AudioSynthesizer] = None


def get_synthesizer() -> AudioSynthesizer:
    """Get the global AudioSynthesizer instance"""
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = AudioSynthesizer()
    return _synthesizer_instance
