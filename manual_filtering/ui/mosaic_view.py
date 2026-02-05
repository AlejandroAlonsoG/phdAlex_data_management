"""
Mosaic view component for displaying images in a paginated grid.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Callable, Optional
from PIL import Image, ImageTk
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import gc
import os

from .image_utils import load_image_as_thumbnail, get_image_info


class LRUCache:
    """
    Limited-size LRU cache for thumbnails to prevent memory exhaustion.
    Automatically evicts oldest entries when max size is reached.
    """
    
    def __init__(self, max_size: int = 400):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key):
        """Get item and move to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Add item, evicting oldest if at capacity."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest entries (remove 25% to avoid frequent evictions)
                num_to_remove = max(1, self.max_size // 4)
                for _ in range(num_to_remove):
                    if self.cache:
                        self.cache.popitem(last=False)
            self.cache[key] = value
    
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        return self.get(key)
    
    def pop(self, key, default=None):
        return self.cache.pop(key, default)
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
    
    def __len__(self):
        return len(self.cache)


class MosaicView(ttk.Frame):
    """
    A paginated mosaic/grid view of images with selection capability.
    """
    
    def __init__(
        self, 
        parent, 
        images_per_page: int = 200,
        thumbnail_size: tuple = (120, 120),
        on_delete_callback: Optional[Callable[[List[str]], None]] = None,
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.images_per_page = images_per_page
        self.thumbnail_size = thumbnail_size
        self.on_delete_callback = on_delete_callback
        
        # State
        self.all_images: List[str] = []
        self.current_page = 0
        self.total_pages = 0
        self.selected_images: set = set()
        # LRU cache limited to ~2 pages worth of thumbnails to prevent memory issues
        self.thumbnail_cache = LRUCache(max_size=images_per_page * 2)
        self.image_widgets: Dict[str, ttk.Frame] = {}
        self.loading = False
        
        # Thread pool for thumbnail loading (reduced workers to limit concurrent memory use)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the mosaic view UI."""
        # Top toolbar
        self.toolbar = ttk.Frame(self)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Back button
        self.back_btn = ttk.Button(self.toolbar, text="← Back to Folders", command=self._on_back)
        self.back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Current folder label
        self.folder_label = ttk.Label(self.toolbar, text="", font=('Segoe UI', 10, 'bold'))
        self.folder_label.pack(side=tk.LEFT, padx=5)
        
        # Selection info
        self.selection_label = ttk.Label(self.toolbar, text="Selected: 0")
        self.selection_label.pack(side=tk.RIGHT, padx=5)
        
        # Delete button
        self.delete_btn = ttk.Button(
            self.toolbar, 
            text="🗑 Delete Selected", 
            command=self._on_delete,
            state=tk.DISABLED
        )
        self.delete_btn.pack(side=tk.RIGHT, padx=5)
        
        # Select all / Deselect all buttons
        self.select_all_btn = ttk.Button(self.toolbar, text="Select All (Page)", command=self._select_all_page)
        self.select_all_btn.pack(side=tk.RIGHT, padx=2)
        
        self.deselect_all_btn = ttk.Button(self.toolbar, text="Deselect All", command=self._deselect_all)
        self.deselect_all_btn.pack(side=tk.RIGHT, padx=2)
        
        # Scrollable canvas for images
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='#2b2b2b', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas to hold image grid
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor=tk.NW)
        
        # Bind resize and scroll events
        self.grid_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        
        # Pagination controls
        self.pagination_frame = ttk.Frame(self)
        self.pagination_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.prev_btn = ttk.Button(self.pagination_frame, text="← Previous", command=self._prev_page)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.page_label = ttk.Label(self.pagination_frame, text="Page 0 of 0")
        self.page_label.pack(side=tk.LEFT, expand=True)
        
        self.images_label = ttk.Label(self.pagination_frame, text="0 images")
        self.images_label.pack(side=tk.LEFT, expand=True)
        
        self.next_btn = ttk.Button(self.pagination_frame, text="Next →", command=self._next_page)
        self.next_btn.pack(side=tk.RIGHT, padx=5)
        
        # Loading indicator
        self.loading_label = ttk.Label(self, text="Loading...", font=('Segoe UI', 12))
        
        # Back callback (set by parent)
        self.on_back_callback: Optional[Callable] = None
    
    def _on_frame_configure(self, event):
        """Update scroll region when frame changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Adjust grid frame width to match canvas."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if self.winfo_ismapped():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def set_images(self, images: List[str], folder_name: str = ""):
        """Set the list of images to display."""
        # Clear cache when switching folders to free memory
        self._clear_cache()
        
        self.all_images = images
        self.current_page = 0
        self.total_pages = max(1, (len(images) + self.images_per_page - 1) // self.images_per_page)
        self.selected_images.clear()
        self.folder_label.config(text=folder_name)
        self._update_selection_label()
        self._load_current_page()
    
    def _clear_cache(self):
        """Clear thumbnail cache and force garbage collection."""
        self.thumbnail_cache.clear()
        gc.collect()
    
    def _load_current_page(self):
        """Load and display thumbnails for the current page."""
        if self.loading:
            return
        
        self.loading = True
        self._show_loading()
        
        # Clear existing widgets
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.image_widgets.clear()
        
        # Calculate page range
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.all_images))
        page_images = self.all_images[start_idx:end_idx]
        
        # Update pagination labels
        self.page_label.config(text=f"Page {self.current_page + 1} of {self.total_pages}")
        self.images_label.config(text=f"{len(self.all_images)} images total")
        
        # Enable/disable pagination buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < self.total_pages - 1 else tk.DISABLED)
        
        # Load thumbnails in background
        def load_thumbnails():
            thumbnails = []
            for filepath in page_images:
                if filepath not in self.thumbnail_cache:
                    thumb = load_image_as_thumbnail(filepath, self.thumbnail_size)
                    if thumb:
                        thumbnails.append((filepath, thumb))
                else:
                    thumbnails.append((filepath, None))  # Already cached
            
            # Update UI in main thread
            self.after(0, lambda: self._display_thumbnails(page_images, thumbnails))
        
        self.executor.submit(load_thumbnails)
    
    def _display_thumbnails(self, page_images: List[str], thumbnails: List[tuple]):
        """Display thumbnails in the grid (called from main thread)."""
        # Convert PIL images to PhotoImage and cache with LRU eviction
        for filepath, pil_image in thumbnails:
            if pil_image is not None and filepath not in self.thumbnail_cache:
                try:
                    photo = ImageTk.PhotoImage(pil_image)
                    self.thumbnail_cache.put(filepath, photo)
                except Exception as e:
                    # If we still hit memory issues, clear cache and retry
                    print(f"Memory warning, clearing cache: {e}")
                    self._clear_cache()
                    try:
                        photo = ImageTk.PhotoImage(pil_image)
                        self.thumbnail_cache.put(filepath, photo)
                    except Exception:
                        pass  # Skip this thumbnail
                # Close PIL image to free memory
                pil_image.close()
        
        # Calculate grid dimensions based on canvas width
        canvas_width = self.canvas.winfo_width()
        if canvas_width < 100:
            canvas_width = 800  # Default if not yet rendered
        
        thumb_width = self.thumbnail_size[0] + 20  # Include padding
        columns = max(1, canvas_width // thumb_width)
        
        # Create image tiles
        for idx, filepath in enumerate(page_images):
            row = idx // columns
            col = idx % columns
            
            self._create_image_tile(filepath, row, col)
        
        self._hide_loading()
        self.loading = False
    
    def _create_image_tile(self, filepath: str, row: int, col: int):
        """Create a single image tile with selection capability."""
        # Frame for the tile
        tile_frame = ttk.Frame(self.grid_frame, padding=3)
        tile_frame.grid(row=row, column=col, padx=3, pady=3, sticky='nsew')
        
        # Is selected?
        is_selected = filepath in self.selected_images
        
        # Image container with border for selection indication
        border_color = '#00aaff' if is_selected else '#3a3a3a'
        img_container = tk.Frame(tile_frame, bg=border_color, padx=3, pady=3)
        img_container.pack()
        
        # Image label
        if filepath in self.thumbnail_cache:
            photo = self.thumbnail_cache[filepath]
            img_label = tk.Label(img_container, image=photo, bg='#2b2b2b')
            img_label.image = photo  # Keep reference
        else:
            img_label = tk.Label(img_container, text="?", width=15, height=8, bg='#2b2b2b', fg='white')
        
        img_label.pack()
        
        # Filename label (truncated)
        info = get_image_info(filepath)
        filename = info['filename']
        if len(filename) > 18:
            filename = filename[:15] + "..."
        
        name_label = ttk.Label(tile_frame, text=filename, font=('Segoe UI', 8))
        name_label.pack()
        
        # Size label
        size_label = ttk.Label(tile_frame, text=info['size_formatted'], font=('Segoe UI', 7))
        size_label.pack()
        
        # Bind click events for selection
        for widget in [tile_frame, img_container, img_label, name_label, size_label]:
            widget.bind('<Button-1>', lambda e, fp=filepath: self._toggle_selection(fp))
            widget.bind('<Control-Button-1>', lambda e, fp=filepath: self._toggle_selection(fp))
        
        # Store reference
        self.image_widgets[filepath] = tile_frame
    
    def _toggle_selection(self, filepath: str):
        """Toggle selection state of an image."""
        if filepath in self.selected_images:
            self.selected_images.remove(filepath)
        else:
            self.selected_images.add(filepath)
        
        self._update_tile_selection(filepath)
        self._update_selection_label()
    
    def _update_tile_selection(self, filepath: str):
        """Update the visual selection state of a tile."""
        if filepath not in self.image_widgets:
            return
        
        tile = self.image_widgets[filepath]
        is_selected = filepath in self.selected_images
        
        # Find the image container and update border
        for child in tile.winfo_children():
            if isinstance(child, tk.Frame):
                child.configure(bg='#00aaff' if is_selected else '#3a3a3a')
                break
    
    def _update_selection_label(self):
        """Update the selection count label."""
        count = len(self.selected_images)
        self.selection_label.config(text=f"Selected: {count}")
        self.delete_btn.config(state=tk.NORMAL if count > 0 else tk.DISABLED)
    
    def _select_all_page(self):
        """Select all images on current page."""
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.all_images))
        
        for filepath in self.all_images[start_idx:end_idx]:
            self.selected_images.add(filepath)
            self._update_tile_selection(filepath)
        
        self._update_selection_label()
    
    def _deselect_all(self):
        """Deselect all images."""
        for filepath in list(self.selected_images):
            self.selected_images.remove(filepath)
            self._update_tile_selection(filepath)
        
        self._update_selection_label()
    
    def _on_delete(self):
        """Handle delete button click."""
        count = len(self.selected_images)
        if count == 0:
            return
        
        # Confirmation dialog
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Move {count} selected image(s) to the Recycle Bin?\n\nThis action can be undone from the Recycle Bin.",
            icon='warning'
        )
        
        if result and self.on_delete_callback:
            files_to_delete = list(self.selected_images)
            self.on_delete_callback(files_to_delete)
            
            # Remove deleted files from our lists
            for filepath in files_to_delete:
                if filepath in self.all_images:
                    self.all_images.remove(filepath)
                self.selected_images.discard(filepath)
                self.thumbnail_cache.pop(filepath, None)
            
            # Recalculate pages and refresh
            self.total_pages = max(1, (len(self.all_images) + self.images_per_page - 1) // self.images_per_page)
            if self.current_page >= self.total_pages:
                self.current_page = max(0, self.total_pages - 1)
            
            self._update_selection_label()
            self._load_current_page()
    
    def _prev_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._load_current_page()
    
    def _next_page(self):
        """Go to next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._load_current_page()
    
    def _on_back(self):
        """Handle back button click."""
        # Clear cache when going back to free memory
        self._clear_cache()
        if self.on_back_callback:
            self.on_back_callback()
    
    def _show_loading(self):
        """Show loading indicator."""
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def _hide_loading(self):
        """Hide loading indicator."""
        self.loading_label.place_forget()
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        self._clear_cache()
