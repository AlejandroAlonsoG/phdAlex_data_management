"""
Main window for the manual image filtering application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List
import os
from pathlib import Path

from send2trash import send2trash

from .image_utils import get_first_level_subdirs, get_all_images_recursive
from .mosaic_view import MosaicView


class MainWindow:
    """Main application window."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manual Image Filtering")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure dark theme
        self._setup_theme()
        
        # State
        self.current_directory: Optional[str] = None
        self.subdirs: List[tuple] = []
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Views
        self.folder_view: Optional[ttk.Frame] = None
        self.mosaic_view: Optional[MosaicView] = None
        
        # Start with folder selection view
        self._show_folder_view()
    
    def _setup_theme(self):
        """Configure a dark theme for the application."""
        style = ttk.Style()
        
        # Use clam as base theme for better customization
        style.theme_use('clam')
        
        # Colors
        bg_dark = '#1e1e1e'
        bg_medium = '#2d2d2d'
        bg_light = '#3a3a3a'
        fg_main = '#ffffff'
        fg_dim = '#aaaaaa'
        accent = '#0078d4'
        
        # Configure styles
        style.configure('TFrame', background=bg_dark)
        style.configure('TLabel', background=bg_dark, foreground=fg_main)
        style.configure('TButton', background=bg_light, foreground=fg_main, padding=8)
        style.map('TButton',
            background=[('active', accent), ('pressed', accent)],
            foreground=[('active', fg_main), ('pressed', fg_main)]
        )
        
        style.configure('Folder.TButton', padding=15, font=('Segoe UI', 11))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('SubHeader.TLabel', font=('Segoe UI', 10), foreground=fg_dim)
        
        # Configure root window
        self.root.configure(bg=bg_dark)
    
    def _show_folder_view(self):
        """Show the folder selection/navigation view."""
        # Hide mosaic view if visible
        if self.mosaic_view:
            self.mosaic_view.pack_forget()
        
        # Create folder view if needed
        if not self.folder_view:
            self.folder_view = ttk.Frame(self.main_frame)
            self._build_folder_view()
        
        self.folder_view.pack(fill=tk.BOTH, expand=True)
    
    def _build_folder_view(self):
        """Build the folder selection view UI."""
        # Header section
        header_frame = ttk.Frame(self.folder_view)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = ttk.Label(
            header_frame, 
            text="Manual Image Filtering",
            style='Header.TLabel'
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Select a root directory to begin, then choose a subdirectory to view images",
            style='SubHeader.TLabel'
        )
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Directory selection section
        dir_frame = ttk.Frame(self.folder_view)
        dir_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.select_dir_btn = ttk.Button(
            dir_frame,
            text="📁 Select Root Directory",
            command=self._select_directory
        )
        self.select_dir_btn.pack(side=tk.LEFT)
        
        self.current_dir_label = ttk.Label(dir_frame, text="No directory selected")
        self.current_dir_label.pack(side=tk.LEFT, padx=20)
        
        # Subdirectories section
        subdirs_label = ttk.Label(
            self.folder_view,
            text="First-Level Subdirectories:",
            style='Header.TLabel'
        )
        subdirs_label.pack(anchor=tk.W, padx=20, pady=(20, 10))
        
        # Scrollable frame for subdirectories
        subdirs_container = ttk.Frame(self.folder_view)
        subdirs_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Canvas with scrollbar
        self.subdirs_canvas = tk.Canvas(subdirs_container, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(subdirs_container, orient=tk.VERTICAL, command=self.subdirs_canvas.yview)
        
        self.subdirs_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.subdirs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for subdirectory buttons
        self.subdirs_frame = ttk.Frame(self.subdirs_canvas)
        self.subdirs_canvas_window = self.subdirs_canvas.create_window(
            (0, 0), 
            window=self.subdirs_frame, 
            anchor=tk.NW
        )
        
        # Bind events
        self.subdirs_frame.bind('<Configure>', self._on_subdirs_frame_configure)
        self.subdirs_canvas.bind('<Configure>', self._on_subdirs_canvas_configure)
        self.subdirs_canvas.bind_all('<MouseWheel>', self._on_subdirs_mousewheel)
        
        # Info label when no subdirs
        self.no_subdirs_label = ttk.Label(
            self.subdirs_frame,
            text="Select a root directory to see subdirectories here",
            style='SubHeader.TLabel'
        )
        self.no_subdirs_label.pack(pady=50)
    
    def _on_subdirs_frame_configure(self, event):
        """Update scroll region."""
        self.subdirs_canvas.configure(scrollregion=self.subdirs_canvas.bbox("all"))
    
    def _on_subdirs_canvas_configure(self, event):
        """Adjust frame width to canvas."""
        self.subdirs_canvas.itemconfig(self.subdirs_canvas_window, width=event.width)
    
    def _on_subdirs_mousewheel(self, event):
        """Handle mousewheel scroll."""
        if self.folder_view.winfo_ismapped():
            self.subdirs_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _select_directory(self):
        """Open directory selection dialog."""
        directory = filedialog.askdirectory(
            title="Select Root Directory",
            mustexist=True
        )
        
        if directory:
            self.current_directory = directory
            self.current_dir_label.config(text=directory)
            self._load_subdirectories()
    
    def _load_subdirectories(self):
        """Load and display first-level subdirectories."""
        if not self.current_directory:
            return
        
        # Clear existing buttons
        for widget in self.subdirs_frame.winfo_children():
            widget.destroy()
        
        # Get subdirectories
        self.subdirs = get_first_level_subdirs(self.current_directory)
        
        if not self.subdirs:
            no_subdirs = ttk.Label(
                self.subdirs_frame,
                text="No subdirectories found in the selected directory",
                style='SubHeader.TLabel'
            )
            no_subdirs.pack(pady=50)
            return
        
        # Create a button for each subdirectory
        for full_path, folder_name in self.subdirs:
            # Count images in this subdirectory (recursive)
            images = get_all_images_recursive(full_path)
            image_count = len(images)
            
            # Create button frame
            btn_frame = ttk.Frame(self.subdirs_frame)
            btn_frame.pack(fill=tk.X, pady=2)
            
            # Folder button
            btn_text = f"📁 {folder_name}"
            btn = ttk.Button(
                btn_frame,
                text=btn_text,
                style='Folder.TButton',
                command=lambda fp=full_path, fn=folder_name: self._open_subdirectory(fp, fn)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Image count label
            count_label = ttk.Label(
                btn_frame,
                text=f"{image_count} images",
                style='SubHeader.TLabel'
            )
            count_label.pack(side=tk.RIGHT, padx=10)
    
    def _open_subdirectory(self, path: str, name: str):
        """Open a subdirectory in the mosaic view."""
        # Get all images recursively
        images = get_all_images_recursive(path)
        
        if not images:
            messagebox.showinfo(
                "No Images",
                f"No supported images found in '{name}' or its subdirectories."
            )
            return
        
        # Hide folder view
        self.folder_view.pack_forget()
        
        # Create mosaic view if needed
        if not self.mosaic_view:
            self.mosaic_view = MosaicView(
                self.main_frame,
                images_per_page=200,
                thumbnail_size=(120, 120),
                on_delete_callback=self._delete_images
            )
            self.mosaic_view.on_back_callback = self._show_folder_view
        
        # Show mosaic view with images
        self.mosaic_view.pack(fill=tk.BOTH, expand=True)
        self.mosaic_view.set_images(images, f"{name} ({len(images)} images)")
    
    def _delete_images(self, filepaths: List[str]):
        """Delete images by moving them to the recycle bin."""
        deleted_count = 0
        errors = []
        
        for filepath in filepaths:
            try:
                send2trash(filepath)
                deleted_count += 1
            except Exception as e:
                errors.append(f"{os.path.basename(filepath)}: {str(e)}")
        
        # Show result
        if errors:
            error_msg = "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... and {len(errors) - 10} more errors"
            
            messagebox.showwarning(
                "Delete Completed with Errors",
                f"Moved {deleted_count} file(s) to Recycle Bin.\n\n"
                f"Failed to delete {len(errors)} file(s):\n{error_msg}"
            )
        else:
            messagebox.showinfo(
                "Delete Completed",
                f"Successfully moved {deleted_count} file(s) to the Recycle Bin."
            )
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
    
    def cleanup(self):
        """Clean up resources before closing."""
        if self.mosaic_view:
            self.mosaic_view.cleanup()
