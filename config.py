import sys
import os
from pathlib import Path

def get_base_path():
    """Get the correct base path for both development and packaged versions"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    else:
        # Running in development
        return Path(__file__).parent

# Define important paths
BASE_DIR = get_base_path()
RECORDINGS_DIR = BASE_DIR / "recordings"
LIBRARY_DIR = BASE_DIR / "library"

# Create directories if they don't exist
RECORDINGS_DIR.mkdir(exist_ok=True)
LIBRARY_DIR.mkdir(exist_ok=True)