#!/usr/bin/env python3
"""
AI-Powered Fantasy Football Draft Assistant
Main entry point for the application
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_app import main

if __name__ == "__main__":
    main()
