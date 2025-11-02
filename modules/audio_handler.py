# modules/audio_handler.py: Multilingual audio handler for Nigerian languages
import os
import uuid
import tempfile
import logging
import speech_recognition as sr
from gtts import gTTS

from modules.language_utils import detect_language, is_pidgin, get_language_name

logger = logging.getLogger(__name__)

# N-ATLAS setup for Nigerian languages
try:
    from transformers import pipeline
    import librosa
    
    # Initialize N-ATLAS ASR pipelines for Nigerian languages
    NATLAS_MODELS = {
        "ha": "NCAIR1/Hausa-ASR",
        "ig": "NCAIR1/Igbo-ASR",
        "yo": "NCAIR1/Yoruba-ASR"
    }
    
    NATLAS_PIPELINES = {}
    for lang_code, model_name in NATLAS_MODELS.items():
        try:
            NATLAS_PIPELINES[lang_code] = pipeline("automatic-speech-recognition", model=model_name)
            logger.info(f"N-ATLAS {lang_code.upper()} model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load N-ATLAS {lang_code.upper()} model: {e}")
    
    USE_NATLAS = len(NATLAS_PIPELINES) > 0
    logger.info(f"N-ATLAS available for languages: {list(NATLAS_PIPELINES.keys())}")
except Exception as e:
    USE_NATLAS = False
    NATLAS_PIPELINES = {}
    logger.warning(f"N-ATLAS not available: {e}")

# Whisper setup
try:
    import whisper
    WHISPER_MODEL = whisper.load_model("base")  # Better accuracy than tiny
    USE_WHISPER = True
    logger.info("Whisper base model loaded for multilingual speech recognition.")
except Exception as e:
    USE_WHISPER = False
    logger.warning(f"Whisper not available: {e}")


def speech_to_text(audio_file_path: str) -> str:
    """
    Efficient multilingual speech-to-text for Nigerian languages.
    Priority: N-ATLAS -> Whisper -> Google Speech Recognition
    First detects language, then transcribes using the best available method.
    """
    if not os.path.isfile(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return "Audio file not found. Please check the path."

    try:
        # Step 1: Language Detection using Whisper
        detected_lang = None
        if USE_WHISPER:
            logger.info("Detecting language from audio...")
            try:
                # Use Whisper to detect language first
                result = WHISPER_MODEL.transcribe(
                    audio_file_path,
                    fp16=False,
                    verbose=False,
                    language=None  # Auto-detect language
                )
                
                # Get detected language
                detected_lang_info = result.get("language", "en")
                
                # Map Whisper language codes to our codes
                lang_mapping = {
                    "yo": "yo",
                    "ig": "ig", 
                    "ha": "ha",
                    "en": "en",
                }
                
                detected_lang = lang_mapping.get(detected_lang_info, "en")
                initial_text = result.get("text", "").strip()
                
                # Additional check for Pidgin
                if detected_lang == "en" and initial_text:
                    if is_pidgin(initial_text):
                        detected_lang = "pidgin"
                
                logger.info(f"Detected language: {detected_lang} ({get_language_name(detected_lang)})")
                    
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                detected_lang = "en"  # Default to English
        
        # Step 2: Try N-ATLAS for Nigerian languages (Hausa, Igbo, Yoruba)
        if detected_lang and detected_lang in NATLAS_PIPELINES:
            try:
                logger.info(f"Attempting N-ATLAS transcription for {get_language_name(detected_lang)}...")
                
                # Load audio file at 16kHz (recommended for N-ATLAS)
                audio, sr = librosa.load(audio_file_path, sr=16000)
                
                # Transcribe using N-ATLAS
                asr_pipeline = NATLAS_PIPELINES[detected_lang]
                result = asr_pipeline(audio)
                text = result.get("text", "").strip()
                
                if text and len(text) > 2:
                    logger.info(f"N-ATLAS transcription successful: '{text[:50]}...'")
                    return text
                else:
                    logger.warning("N-ATLAS returned empty or very short transcription")
                    
            except Exception as e:
                logger.warning(f"N-ATLAS transcription failed: {e}")
        
        # Step 3: Try Whisper as fallback for targeted transcription
        if detected_lang:
            # Try targeted transcription with detected language
            if USE_WHISPER and detected_lang in ["yo", "ig", "ha", "en"]:
                try:
                    logger.info(f"Transcribing audio with Whisper as {get_language_name(detected_lang)}...")
                    
                    # For Nigerian languages, try specific language hint
                    if detected_lang != "en":
                        result = WHISPER_MODEL.transcribe(
                            audio_file_path,
                            language=detected_lang,
                            fp16=False,
                            verbose=False,
                        )
                    else:
                        # For English/Pidgin, use English setting
                        result = WHISPER_MODEL.transcribe(
                            audio_file_path,
                            language="en",
                            fp16=False,
                            verbose=False,
                        )
                    
                    text = result.get("text", "").strip()
                    if text and len(text) > 2:
                        logger.info(f"Whisper transcription successful: '{text[:50]}...'")
                        return text
                        
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
            
            # Step 4: Fallback to Google Speech Recognition with detected language
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                
                with sr.AudioFile(audio_file_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                
                # Map to Google language codes
                google_lang_map = {
                    "yo": "yo",
                    "ig": "ig",
                    "ha": "ha",
                    "en": "en-NG",  # Nigerian English
                    "pidgin": "en-NG"  # Use Nigerian English for Pidgin
                }
                
                google_lang = google_lang_map.get(detected_lang, "en-NG")
                logger.info(f"Trying Google Speech Recognition with {google_lang}...")
                
                text = r.recognize_google(audio_data, language=google_lang)
                if text and len(text) > 2:
                    logger.info(f"Google transcription successful: '{text[:50]}...'")
                    return text
                    
            except sr.UnknownValueError:
                logger.warning("Google could not understand the audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition error: {e}")
            except Exception as e:
                logger.error(f"Google Speech setup failed: {e}")
        
        # Step 5: Final fallback - try general auto-detection with Whisper
        if USE_WHISPER:
            try:
                logger.info("Attempting final Whisper transcription with auto-detection...")
                result = WHISPER_MODEL.transcribe(
                    audio_file_path,
                    fp16=False,
                    verbose=False,
                )
                text = result.get("text", "").strip()
                if text and len(text) > 2:
                    logger.info(f"Final Whisper transcription successful: '{text[:50]}...'")
                    return text
            except Exception as e:
                logger.error(f"Final transcription attempt failed: {e}")
        
        return "Could not transcribe audio. Please speak clearly and ensure good audio quality."
        
    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        return "Error processing audio. Please try again."


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Multilingual text-to-speech supporting Nigerian languages.
    Produces complete, well-paced audio with Nigerian accent where possible.
    """
    try:
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided for TTS")

        # Don't truncate - allow full response
        logger.info(f"Creating TTS for language: {lang} ({get_language_name(lang)})")

        tts_config = get_tts_config(text, lang)

        try:
            # Use slow=False for normal pace, slow=True for slower pace if needed
            tts = gTTS(
                text=text,
                lang=tts_config["lang"],
                tld=tts_config["tld"],
                slow=False,  # Normal pace for better natural flow
            )
            logger.info(f"Using TTS config: {tts_config}")

        except Exception as e:
            logger.warning(f"Primary TTS config failed: {e}")
            # Fallback to Nigerian English
            tts = gTTS(text=text, lang="en", tld="com.ng", slow=False)
            logger.info("Using Nigerian English fallback for TTS")

        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"tts_{uuid.uuid4().hex}.mp3")

        tts.save(file_path)
        logger.info(f"TTS audio saved: {file_path}")

        return file_path

    except Exception as e:
        logger.error(f"TTS conversion failed: {e}")

        try:
            logger.info("Attempting final TTS fallback...")
            fallback_tts = gTTS(text=text, lang="en", slow=False)
            temp_dir = tempfile.gettempdir()
            fallback_path = os.path.join(
                temp_dir, f"tts_emergency_{uuid.uuid4().hex}.mp3"
            )
            fallback_tts.save(fallback_path)
            return fallback_path

        except Exception as e2:
            logger.error(f"All TTS attempts failed: {e2}")
            raise Exception(f"TTS conversion completely failed: {e}")


def get_tts_config(text: str, lang: str) -> dict:
    """
    Get optimal TTS configuration for each Nigerian language.
    Uses Nigerian English accent (tld='com.ng') for Nigerian context.
    """
    config = {"lang": "en", "tld": "com.ng", "slow": False}

    if lang == "yo":
        # Yoruba has native TTS support
        config = {"lang": "yo", "tld": "com", "slow": False}
        logger.info("Using native Yoruba TTS")

    elif lang == "ig":
        # Igbo - use Nigerian English
        config = {"lang": "en", "tld": "com.ng", "slow": False}
        logger.info("Using Nigerian English for Igbo")

    elif lang == "ha":
        # Hausa has native TTS support
        config = {"lang": "ha", "tld": "com", "slow": False}
        logger.info("Using native Hausa TTS")

    elif lang == "pidgin":
        # Nigerian Pidgin - use Nigerian English
        config = {"lang": "en", "tld": "com.ng", "slow": False}
        logger.info("Using Nigerian English for Pidgin")

    elif lang == "en":
        # English - use Nigerian accent
        config = {"lang": "en", "tld": "com.ng", "slow": False}
        logger.info("Using Nigerian English TTS")

    else:
        # Default to Nigerian English
        config = {"lang": "en", "tld": "com.ng", "slow": False}
        logger.info("Using default Nigerian English TTS")

    return config


def cleanup_audio_file(file_path: str) -> None:
    """
    Clean up temporary audio files.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up audio file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


def get_supported_tts_languages():
    """
    Return list of languages with TTS support.
    """
    return {
        "yo": "Yoruba (Native TTS)",
        "ha": "Hausa (Native TTS)",
        "ig": "Igbo (Nigerian English TTS)",
        "pidgin": "Nigerian Pidgin (Nigerian English TTS)",
        "en": "English (Nigerian accent)",
    }