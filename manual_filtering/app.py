#!/usr/bin/env python3
"""
Manual Image Filtering Application

A desktop application for visually filtering and organizing images from directories.
Supports browsing first-level subdirectories, viewing images in a paginated mosaic,
and batch deleting selected images to the recycle bin.

Usage:
    python app.py

Supported image formats:
    .jpg, .jpeg, .png, .gif, .bmp, .webp, .tif, .tiff,
    .psd, .orf, .nef, .arw, .ai, .eps
"""

import sys
import tkinter as tk
from pathlib import Path

# Add the app directory to path for imports
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from ui.main_window import MainWindow


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow (pip install Pillow)")
    
    try:
        from send2trash import send2trash
    except ImportError:
        missing.append("send2trash (pip install send2trash)")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall them with: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main entry point."""
    # Check dependencies
    check_dependencies()
    
    # Create root window
    root = tk.Tk()
    
    # Set icon if available (optional)
    try:
        # You can add a custom icon here
        pass
    except Exception:
        pass
    
    # Create and run application
    app = MainWindow(root)
    
    # Handle window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the application
    app.run()


if __name__ == "__main__":
    main()
