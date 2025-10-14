"""
This script installs dependencies needed for testing the memory system.
"""

import subprocess
import sys

def install_dependencies():
    """Install necessary packages for testing"""
    print("Installing required packages for memory system testing...")
    
    dependencies = [
        "pytest",
        "pytest-asyncio",
        "redis",
        "fakeredis"  # For mocking Redis in tests
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("Successfully installed dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()