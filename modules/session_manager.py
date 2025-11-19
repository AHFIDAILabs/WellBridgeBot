# modules/session_manager.py: Centralized session state management for Streamlit
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import streamlit as st


@dataclass
class Message:
    """Represents a single chat message"""
    role: str  # "user" or "assistant"
    content: str
    source: Optional[str] = None
    has_audio: bool = False
    is_voice: bool = False
    language: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "has_audio": self.has_audio,
            "is_voice": self.is_voice,
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SessionState:
    """Application session state"""
    messages: List[Message] = field(default_factory=list)
    audio_responses: Dict[str, str] = field(default_factory=dict)
    last_audio_processed: Optional[str] = None
    selected_language: str = "auto"
    conversation_id: Optional[str] = None
    
    def add_user_message(self, content: str, is_voice: bool = False, language: Optional[str] = None):
        """Add a user message to the conversation"""
        self.messages.append(
            Message(
                role="user",
                content=content,
                is_voice=is_voice,
                language=language
            )
        )
    
    def add_assistant_message(
        self,
        content: str,
        source: str,
        has_audio: bool = False,
        language: Optional[str] = None
    ):
        """Add an assistant response to the conversation"""
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                source=source,
                has_audio=has_audio,
                language=language
            )
        )
    
    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None
    
    def clear(self):
        """Clear all session data"""
        self.messages = []
        self.audio_responses = {}
        self.last_audio_processed = None
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation history with optional limit"""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def message_count(self) -> int:
        """Get total number of messages"""
        return len(self.messages)
    
    def store_audio_response(self, message_index: int, audio_path: str):
        """Store audio file path for a message"""
        audio_key = f"audio_{message_index}"
        self.audio_responses[audio_key] = audio_path
    
    def get_audio_response(self, message_index: int) -> Optional[str]:
        """Get audio file path for a message"""
        audio_key = f"audio_{message_index}"
        return self.audio_responses.get(audio_key)


class SessionManager:
    """Manages Streamlit session state with type safety"""
    
    STATE_KEY = "app_state"
    
    @staticmethod
    def get_state() -> SessionState:
        """Get the current session state (creates if doesn't exist)"""
        if SessionManager.STATE_KEY not in st.session_state:
            st.session_state[SessionManager.STATE_KEY] = SessionState()
        return st.session_state[SessionManager.STATE_KEY]
    
    @staticmethod
    def clear_state():
        """Clear the current session state"""
        if SessionManager.STATE_KEY in st.session_state:
            st.session_state[SessionManager.STATE_KEY].clear()
    
    @staticmethod
    def reset_state():
        """Completely reset session state"""
        if SessionManager.STATE_KEY in st.session_state:
            del st.session_state[SessionManager.STATE_KEY]
    
    @staticmethod
    def get_or_create(key: str, default_value=None):
        """Get or create a session state value"""
        if key not in st.session_state:
            st.session_state[key] = default_value
        return st.session_state[key]
    
    @staticmethod
    def set_value(key: str, value):
        """Set a session state value"""
        st.session_state[key] = value
    
    @staticmethod
    def has_key(key: str) -> bool:
        """Check if a key exists in session state"""
        return key in st.session_state
