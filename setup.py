import os
import sys

def ensure_directories():
    """Ensure required directories exist."""
    directories = ['logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
            
    print("Setup complete!")

if __name__ == "__main__":
    ensure_directories()
    
    # Check if pygame is installed
    try:
        import pygame
        print("Pygame is installed successfully.")
    except ImportError:
        print("Warning: Pygame is not installed. Run 'pip install -r requirements.txt' to install dependencies.")
    
    # Check if numpy is installed
    try:
        import numpy
        print("Numpy is installed successfully.")
    except ImportError:
        print("Warning: Numpy is not installed. Run 'pip install -r requirements.txt' to install dependencies.") 