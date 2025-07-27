# Audio handling utilities for Lighthouse HealthConnect: modules/audio_handler.py
import speech_recognition as sr
from gtts import gTTS
import os
import uuid
import tempfile

def speech_to_text(audio_file_path):
    """
    Converts speech from an audio file to text.
    """
    if not os.path.isfile(audio_file_path):
        return "Audio file not found. Please check the path."

    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio. The speech may be unclear or the file may be empty."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}. Please check your network connection."

def text_to_speech(text, lang='en'):
    """
    Converts text to speech and saves it as a temporary audio file.
    Returns the path to the audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_file_path = temp_audio_file.name
        temp_audio_file.close()
        tts.save(audio_file_path)
        return audio_file_path
    except Exception as e:
        print(f"Error in text-to-speech generation: {e}")
        return None