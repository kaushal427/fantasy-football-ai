#!/usr/bin/env python3
"""
Launcher script for the AI-Powered Fantasy Football Draft Assistant
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    app_path = os.path.join("draft-optimizer", "ai_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Application not found at {app_path}")
        print("Please ensure the draft-optimizer folder contains ai_app.py")
        sys.exit(1)
    
    print("🚀 Launching AI-Powered Fantasy Football Draft Assistant...")
    print("📁 Application location:", app_path)
    
    # Change to the draft-optimizer directory and run the app
    os.chdir("draft-optimizer")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ai_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
