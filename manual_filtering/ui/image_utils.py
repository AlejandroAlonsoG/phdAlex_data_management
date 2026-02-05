"""
Image utilities for loading, thumbnail generation, and format handling.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Set
from PIL import Image, ImageDraw, ImageFont
import io

# Supported image formats
SUPPORTED_FORMATS: Set[str] = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',  # Common formats
    '.tif', '.tiff',  # TIFF
    '.psd',  # Photoshop
    '.orf',  # Olympus RAW
    '.nef',  # Nikon RAW
    '.arw',  # Sony RAW
    '.ai',   # Adobe Illustrator
    '.eps',  # Encapsulated PostScript
}

# Formats that PIL can directly open
NATIVE_FORMATS: Set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tif', '.tiff'}

# RAW formats that need special handling
RAW_FORMATS: Set[str] = {'.orf', '.nef', '.arw'}

# Adobe/Vector formats - may have limited support
ADOBE_FORMATS: Set[str] = {'.psd', '.ai', '.eps'}


def is_supported_image(filepath: str) -> bool:
    """Check if a file is a supported image format."""
    ext = Path(filepath).suffix.lower()
    return ext in SUPPORTED_FORMATS


def get_all_images_recursive(directory: str) -> List[str]:
    """
    Recursively find all supported images in a directory and its subdirectories.
    Returns a sorted list of absolute file paths.
    """
    images = []
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return images
    
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if is_supported_image(filepath):
                images.append(filepath)
    
    # Sort by filename for consistent ordering
    images.sort(key=lambda x: x.lower())
    return images


def create_placeholder_thumbnail(size: Tuple[int, int], text: str, bg_color: str = "#3a3a3a") -> Image.Image:
    """Create a placeholder thumbnail with text for unsupported or failed images."""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw text centered
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill="#ffffff")
    
    return img


def load_image_as_thumbnail(filepath: str, size: Tuple[int, int] = (150, 150)) -> Optional[Image.Image]:
    """
    Load an image and create a thumbnail.
    Handles various formats with fallbacks for unsupported ones.
    Returns a PIL Image or None if loading fails completely.
    """
    ext = Path(filepath).suffix.lower()
    
    try:
        # Try native PIL loading first
        if ext in NATIVE_FORMATS:
            img = Image.open(filepath)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            # Convert to RGB if necessary (for consistency)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        
        # Try PSD files - PIL has limited PSD support
        elif ext == '.psd':
            try:
                img = Image.open(filepath)
                img = img.convert('RGB')
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return img
            except Exception:
                return create_placeholder_thumbnail(size, "PSD", "#4a2c6a")
        
        # RAW formats - try to open, fall back to placeholder
        elif ext in RAW_FORMATS:
            try:
                # Try rawpy if available
                import rawpy
                with rawpy.imread(filepath) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                img = Image.fromarray(rgb)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                return img
            except ImportError:
                # rawpy not installed, try PIL
                try:
                    img = Image.open(filepath)
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img
                except Exception:
                    format_name = ext[1:].upper()
                    return create_placeholder_thumbnail(size, format_name, "#2c4a2c")
            except Exception:
                format_name = ext[1:].upper()
                return create_placeholder_thumbnail(size, format_name, "#2c4a2c")
        
        # AI/EPS files - limited support
        elif ext in {'.ai', '.eps'}:
            try:
                img = Image.open(filepath)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
            except Exception:
                format_name = ext[1:].upper()
                return create_placeholder_thumbnail(size, format_name, "#4a4a2c")
        
        # Unknown but in supported list - try generic open
        else:
            try:
                img = Image.open(filepath)
                img.thumbnail(size, Image.Resampling.LANCZOS)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img
            except Exception:
                format_name = ext[1:].upper() if ext else "?"
                return create_placeholder_thumbnail(size, format_name, "#4a2c2c")
    
    except Exception as e:
        # Last resort - create error placeholder
        return create_placeholder_thumbnail(size, "ERR", "#6a2c2c")


def get_first_level_subdirs(directory: str) -> List[Tuple[str, str]]:
    """
    Get all first-level subdirectories of a directory.
    Returns list of tuples: (full_path, folder_name)
    """
    subdirs = []
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return subdirs
    
    for item in directory.iterdir():
        if item.is_dir():
            subdirs.append((str(item), item.name))
    
    # Sort by name
    subdirs.sort(key=lambda x: x[1].lower())
    return subdirs


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_image_info(filepath: str) -> dict:
    """Get basic info about an image file."""
    path = Path(filepath)
    info = {
        'filename': path.name,
        'path': str(path),
        'extension': path.suffix.lower(),
        'size': 0,
        'size_formatted': 'Unknown',
    }
    
    try:
        stat = path.stat()
        info['size'] = stat.st_size
        info['size_formatted'] = format_file_size(stat.st_size)
    except Exception:
        pass
    
    return info
