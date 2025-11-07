from typing import Dict, Optional
import json
import os

class UserPreferencesManager:
    def __init__(self):
        self.preferences_file = "user_preferences.json"
        self.preferences: Dict[str, dict] = self._load_preferences()
        
        # Supported languages with their codes and names
        self.supported_languages = {
            'en': {
                'name': 'English',
                'welcome': 'Welcome! You can change your language anytime by typing "change language"',
                'select': 'Please select your preferred language:'
            },
            'yo': {
                'name': 'Yoruba',
                'welcome': 'E kaabo! E le yi ede pada nigbakugba nipa titẹ "yi ede pada"',
                'select': 'Jọwọ yan ede ti o fẹ:'
            },
            'ha': {
                'name': 'Hausa',
                'welcome': 'Barka da zuwa! Kuna iya canza harshe a kowane lokaci ta hanyar rubuta "canza harshe"',
                'select': 'Da fatan za a zaɓi harshen da kuke so:'
            },
            'ig': {
                'name': 'Igbo',
                'welcome': 'Ndewo! Ị nwere ike ịgbanwe asụsụ gị mgbe ọ bụla site na ịtịpụ "change language"',
                'select': 'Biko họrọ asụsụ ị chọrọ:'
            },
            'pcm': {
                'name': 'Nigerian Pidgin',
                'welcome': 'You don land! You fit change your language anytime if you type "change language"',
                'select': 'Abeg select the language wey you want:'
            }
        }
    
    def _load_preferences(self) -> dict:
        """Load user preferences from file"""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_preferences(self):
        """Save user preferences to file"""
        with open(self.preferences_file, 'w') as f:
            json.dump(self.preferences, f)
    
    def get_user_language(self, phone_number: str) -> str:
        """Get user's preferred language"""
        return self.preferences.get(phone_number, {}).get('language', 'en')
    
    def set_user_language(self, phone_number: str, language: str):
        """Set user's preferred language"""
        if language in self.supported_languages:
            if phone_number not in self.preferences: 
                self.preferences[phone_number] = {}
            self.preferences[phone_number]['language'] = language
            self._save_preferences()
    
    def get_language_selection_message(self, phone_number: str) -> str:
        """Generate language selection message"""
        current_lang = self.get_user_language(phone_number)
        message = f"{self.supported_languages[current_lang]['select']}\n\n"
        
        # Add numbered list of languages
        for i, (code, lang) in enumerate(self.supported_languages.items(), 1):
            message += f"{i}. {lang['name']} ({code})\n"
        
        return message
    
    def handle_language_selection(self, phone_number: str, message: str) -> Optional[str]:
        """Handle language selection input"""
        # Check if message is a number corresponding to a language
        try:
            selection = int(message.strip())
            if 1 <= selection <= len(self.supported_languages):
                lang_code = list(self.supported_languages.keys())[selection - 1]
                self.set_user_language(phone_number, lang_code)
                return self.supported_languages[lang_code]['welcome']
        except ValueError:
            pass
        
        # Check if message is a language code
        message = message.lower()
        if message in self.supported_languages:
            self.set_user_language(phone_number, message)
            return self.supported_languages[message]['welcome']
        
        return None

    def is_language_change_request(self, message: str) -> bool:
        """Check if message is requesting language change"""
        change_commands = [
            "change language", "yi ede pada", "canza harshe",
            "choose language", "select language", "language"
        ]
        return message.lower().strip() in change_commands