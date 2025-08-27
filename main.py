# Main entry point for the application: main.py
import subprocess
import os

def run_app():
    """
    Runs the Streamlit application.
    """
    # Get the path to the app.py file
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Command to run streamlit
    command = ["streamlit", "run", app_path]
    
    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    run_app()