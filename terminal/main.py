#!/usr/bin/env python3
"""
Main entry point for LumosTrade Terminal
Run this file to start the terminal interface
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terminal.cli import main

if __name__ == "__main__":
    main()
