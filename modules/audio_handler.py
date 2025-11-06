# modules/audio_handler.py: Multilingual audio handler for Nigerian languages
import os
import uuid
import tempfile
import logging
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from transformers import AutoTokenizer, VitsModel, pipeline
import librosa
from dotenv import load_dotenv
import torch
import torchaudio


from modules.language_utils import detect_language, is_pidgin, get_language_name

logger = logging.getLogger(__name__)

# # N-ATLAS setup for Nigerian languages
try:
   
    # Load environment variables to get HuggingFace token
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if hf_token:
        logger.info("HuggingFace token found, will use for N-ATLAS model downloads")
        # Set environment variable for transformers library
        os.environ["HF_TOKEN"] = hf_token
    else:
        logger.warning("HuggingFace token not found in .env file")
    
    # Initialize N-ATLAS ASR pipelines for Nigerian languages
    NATLAS_MODELS = {
        "ha": "Hausa-ASR",
        "ig": "Igbo-ASR",
        "yo": "Yoruba-ASR"
    }


    ## to get current language seesion
     
    NATLAS_PIPELINES = {}
    for lang_code, model_name in NATLAS_MODELS.items():
        try:
            logger.info(f"Loading N-ATLAS {model_name.upper().split('-')[0]} model: {'NCAIR1/'+model_name}...")
            # Pass token explicitly to pipeline
            NATLAS_PIPELINES[lang_code] = pipeline(
                "automatic-speech-recognition", 
                model='NCAIR1/'+model_name,
                token=hf_token if hf_token else None
            )
            logger.info(f"✓ N-ATLAS {model_name.upper().split('-')[0]} model loaded successfully")
        except Exception as e:
            logger.warning(f"✗ Could not load N-ATLAS {model_name.upper().split('-')[0]} model: {e}")
    
    USE_NATLAS = len(NATLAS_PIPELINES) > 0
    if USE_NATLAS:
        logger.info(f"✓ N-ATLAS enabled for languages: {list(NATLAS_PIPELINES.keys())}")
    else:
        logger.warning("✗ N-ATLAS not available - no models loaded successfully")
except Exception as e:
    USE_NATLAS = False
    NATLAS_PIPELINES = {}
    logger.warning(f"✗ N-ATLAS initialization failed: {e}")


def speech_to_text(audio_file_path: str, selected_lang: str):
    """
    Efficient multilingual speech-to-text for Nigerian languages.
    Priority: N-ATLAS -> Whisper -> Google Speech Recognition
    First detects language, then transcribes using the best available method.
    """
    text = ""
    if not os.path.isfile(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return "Audio file not found. Please check the path."

    if selected_lang == "auto":
        try:
            import whisper
            WHISPER_MODEL = whisper.load_model("base")  # Better accuracy than tiny
            USE_WHISPER = True
            logger.info("Whisper base model loaded for multilingual speech recognition.")
        except Exception as e:
            USE_WHISPER = False
            logger.warning(f"Whisper not available: {e}")
    
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

            result_lang = result.get("language")

            if result_lang not in ["ha", "yo", "ig", "en", "pidgin", "hau", "ibo", "yor"]:
                st.error(f"Detected language '{result_lang}' not supported, defaulting to English.\n Please try selecting the language manually next time.")
            return result.get("text", "").strip(), result_lang
        except Exception as e:
            logger.error(f"Whisper language detection failed: {e}")

    elif selected_lang in ['ha', 'ig', 'yo']:
        try:
            ## we will use auto detect to detect the language
            logger.info(f"Attempting N-ATLAS transcription for {selected_lang}...")
            
            # Load audio file at 16kHz (recommended for N-ATLAS)
            audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
            
            # Transcribe using N-ATLAS
            asr_pipeline = NATLAS_PIPELINES[selected_lang]
            result = asr_pipeline(audio_data)
            text = result.get("text", "").strip()

            if text and len(text) > 2:
                logger.info(f"N-ATLAS ({selected_lang}) transcription successful: '{text[:50]}...'")
                detected_lang = selected_lang  # Update detected language
                # return text

            else:
                logger.warning(f"N-ATLAS ({selected_lang}) returned empty or very short transcription")

            return text, selected_lang 

        except Exception as e:
            logger.error(f"Transcription failed: {e}")

    try:
        # Initialize recognizer
        r = sr.Recognizer()

        r.energy_threshold = 300
        r.dynamic_energy_threshold = True

        # Load your audio
        with sr.AudioFile(audio_file_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.record(source)
        print(audio_file_path
                )
        # Try recognizing with multiple language options
        text_g = r.recognize_google(
            audio,
            language=selected_lang
        )
        logger.info(f"Google Speech Recognition transcription successful: '{text_g}...'")

        return text_g, selected_lang

    except sr.UnknownValueError:
        logger.warning("Google could not understand the audio OR language not supported")
    except sr.RequestError as e:
        logger.warning(f"Google Speech Recognition error: {e}")
    except Exception as e:
        logger.error(f"Google Speech setup failed: {e}")


def text_to_speech(text: str, lang: str = "en") -> str:
    """
    Multilingual text-to-speech supporting Nigerian languages.
    Produces complete, well-paced audio with Nigerian accent where possible.
    """
    try:
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided for TTS")

        if lang in ["yo", "ha"]:
            logger.info(f"Using MMS TTS for language: {lang} ({get_language_name(lang)})")
            try:
                mms_lang = {
                    "yo": "yor",  # Yoruba
                    "ha": "hau"   # Hausa
                }                
                model = VitsModel.from_pretrained(f"facebook/mms-tts-{mms_lang.get(lang)}")
                tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{mms_lang.get(lang)}")

                inputs = tokenizer(text, return_tensors="pt")

                with torch.no_grad():
                    output = model(**inputs).waveform

                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, f"tts_{uuid.uuid4().hex}.wav")

                torchaudio.save(file_path, output, sample_rate=16000)

                print(f"mms for {lang} was successful")

                return file_path
            except Exception as e:
                logger.warning(f"MMS TTS failed: {e}")
        else:
            try:
                        # Use slow=False for normal pace, slow=True for slower pace if needed
                # Don't truncate - allow full response
                logger.info(f"Creating TTS for language: {lang} ({get_language_name(lang)})")

                tts_config = get_tts_config(text, lang)
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