from modules.language_preferences import UserPreferencesManager

class LanguageSelectionHandler:
    def __init__(self):
        self.preferences_manager = UserPreferencesManager()
        self.users_in_selection = set()  # Track users currently selecting language
        
    def handle_message(self, phone_number: str, message: str) -> tuple:
        """
        Handle incoming messages for language selection
        Returns: (response_message, selected_language, should_process_further)
        """
        # Check if user is currently selecting a language
        if phone_number in self.users_in_selection:
            response = self.preferences_manager.handle_language_selection(phone_number, message)
            if response:
                self.users_in_selection.remove(phone_number)
                return response, self.preferences_manager.get_user_language(phone_number), True
            else:
                return ("Invalid selection. " + 
                       self.preferences_manager.get_language_selection_message(phone_number),
                       None, False)
        
        # Check if user wants to change language
        if self.preferences_manager.is_language_change_request(message):
            self.users_in_selection.add(phone_number)
            return (self.preferences_manager.get_language_selection_message(phone_number),
                   None, False)
        
        # Regular message, return current language
        return None, self.preferences_manager.get_user_language(phone_number), True
    
    def get_user_language(self, phone_number: str) -> str:
        """Get user's current language preference"""
        return self.preferences_manager.get_user_language(phone_number)