#!/usr/bin/env python3
"""
Annotation Validator — Entry Point
===================================

A desktop application for validating annotations produced by the
data ordering pipeline's merge output.

Validates:
  1. That extracted fields match clues visible in original file paths
  2. That duplicates are correctly identified
  3. That annotation fields make sense with the actual image

Usage:
    cd annotation_validator
    pip install -r requirements.txt
    python app.py
"""

import sys
from pathlib import Path

# Ensure the app directory is on the import path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


def check_dependencies():
    """Check that required packages are installed."""
    missing = []

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import openpyxl
    except ImportError:
        missing.append("openpyxl")

    if missing:
        print("╔═══════════════════════════════════════════════╗")
        print("║  Missing required dependencies:               ║")
        for dep in missing:
            print(f"║    - {dep:<40s} ║")
        print("║                                               ║")
        print("║  Install with:                                ║")
        print("║    pip install -r requirements.txt             ║")
        print("╚═══════════════════════════════════════════════╝")
        sys.exit(1)


def _enable_hidpi():
    """Tell Windows this process is DPI-aware so tkinter renders at native
    resolution instead of being bitmap-upscaled (which causes blurry text)."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        pass  # not Windows, or older Windows — ignore silently


def main():
    """Launch the Annotation Validator application."""
    check_dependencies()
    _enable_hidpi()

    import tkinter as tk
    from main_window import MainWindow

    root = tk.Tk()
    # Let Tk honour the system DPI for font scaling
    root.tk.call('tk', 'scaling', root.winfo_fpixels('1i') / 72)
    app = MainWindow(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    app.run()


if __name__ == "__main__":
    main()
