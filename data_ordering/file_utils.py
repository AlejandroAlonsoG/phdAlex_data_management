"""
File system utilities for the data ordering tool.
Handles directory traversal, file classification, metadata extraction, etc.
"""
import os
import shutil
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import extension sets from config
from .config import (
    IMAGE_EXTENSIONS, IMAGE_WITH_METADATA_SUPPORT,
    TEXT_EXTENSIONS, SPREADSHEET_EXTENSIONS, DATABASE_EXTENSIONS,
    COMPRESSED_EXTENSIONS, VIDEO_EXTENSIONS
)

# Try to import PIL for metadata extraction
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class FileType(Enum):
    """Classification of file types."""
    IMAGE = "image"
    TEXT = "text"
    SPREADSHEET = "spreadsheet"
    DATABASE = "database"
    COMPRESSED = "compressed"
    VIDEO = "video"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Information about a file."""
    path: Path
    file_type: FileType
    extension: str
    size_bytes: int
    
    @property
    def name(self) -> str:
        return self.path.name
    
    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class ImageMetadata:
    """Metadata extracted from an image."""
    date_taken: Optional[str] = None  # YYYYMMDD format
    datetime_original: Optional[datetime] = None
    camera_model: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    def has_date(self) -> bool:
        return self.date_taken is not None


class FileClassifier:
    """Classifies files by their extension using config sets."""
    
    def __init__(self):
        """Initialize the classifier using extension sets from config."""
        # Use extension sets from config
        self.image_extensions: Set[str] = IMAGE_EXTENSIONS
        self.image_with_metadata: Set[str] = IMAGE_WITH_METADATA_SUPPORT
        self.text_extensions: Set[str] = TEXT_EXTENSIONS
        self.spreadsheet_extensions: Set[str] = SPREADSHEET_EXTENSIONS
        self.database_extensions: Set[str] = DATABASE_EXTENSIONS
        self.compressed_extensions: Set[str] = COMPRESSED_EXTENSIONS
        self.video_extensions: Set[str] = VIDEO_EXTENSIONS
    
    def classify(self, path: Path) -> FileType:
        """Classify a file by its extension."""
        ext = path.suffix.lower()
        if ext in self.image_extensions:
            return FileType.IMAGE
        elif ext in self.text_extensions:
            return FileType.TEXT
        elif ext in self.spreadsheet_extensions:
            return FileType.SPREADSHEET
        elif ext in self.database_extensions:
            return FileType.DATABASE
        elif ext in self.compressed_extensions:
            return FileType.COMPRESSED
        elif ext in self.video_extensions:
            return FileType.VIDEO
        elif ext:
            return FileType.OTHER
        else:
            return FileType.UNKNOWN
    
    def supports_metadata(self, path: Path) -> bool:
        """Check if the image format supports metadata extraction."""
        return path.suffix.lower() in self.image_with_metadata
    
    def get_file_info(self, path: Path) -> FileInfo:
        """Get information about a file."""
        return FileInfo(
            path=path,
            file_type=self.classify(path),
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size if path.exists() else 0
        )


class DirectoryWalker:
    """
    Walks directories depth-first.
    Yields directories and their contents.
    """
    
    def __init__(self, classifier: FileClassifier = None):
        self.classifier = classifier or FileClassifier()
    
    def walk_depth_first(self, root: Path) -> Iterator[Tuple[Path, List[FileInfo]]]:
        """
        Walk directories depth-first.
        
        Yields:
            Tuple of (directory_path, list of FileInfo for files in that directory)
        """
        root = Path(root)
        if not root.is_dir():
            return
        
        # Get all items in current directory
        try:
            items = list(root.iterdir())
        except PermissionError:
            return
        
        # Separate files and subdirectories
        files = []
        subdirs = []
        
        for item in items:
            if item.is_file():
                files.append(self.classifier.get_file_info(item))
            elif item.is_dir():
                subdirs.append(item)
        
        # First, recurse into subdirectories (depth-first)
        for subdir in sorted(subdirs):
            yield from self.walk_depth_first(subdir)
        
        # Then yield current directory with its files
        yield (root, files)
    
    def count_files_recursive(self, root: Path) -> Dict[FileType, int]:
        """Count files by type recursively."""
        counts = {ft: 0 for ft in FileType}
        
        for _, files in self.walk_depth_first(root):
            for f in files:
                counts[f.file_type] += 1
        
        return counts


class MetadataExtractor:
    """Extracts metadata from image files."""
    
    @staticmethod
    def extract(path: Path) -> ImageMetadata:
        """
        Extract metadata from an image.
        
        Args:
            path: Path to the image file
            
        Returns:
            ImageMetadata object
        """
        metadata = ImageMetadata()
        
        if not HAS_PIL:
            return metadata
        
        try:
            with Image.open(path) as img:
                metadata.width = img.width
                metadata.height = img.height
                
                # Try to get EXIF data
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag == 'DateTimeOriginal':
                            try:
                                # Parse EXIF date format: "YYYY:MM:DD HH:MM:SS"
                                dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                                metadata.datetime_original = dt
                                metadata.date_taken = dt.strftime("%Y%m%d")
                            except (ValueError, TypeError):
                                pass
                        
                        elif tag == 'Model':
                            metadata.camera_model = str(value).strip()
        
        except Exception:
            # Silently fail for unreadable images
            pass
        
        return metadata


class FileOperations:
    """File operations with logging support."""
    
    @staticmethod
    def move_file(source: Path, dest: Path, create_dirs: bool = True) -> bool:
        """
        Move a file from source to destination.
        
        Args:
            source: Source file path
            dest: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if successful
        """
        try:
            if create_dirs:
                dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            return True
        except Exception:
            return False
    
    @staticmethod
    def copy_file(source: Path, dest: Path, create_dirs: bool = True) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            source: Source file path
            dest: Destination file path
            create_dirs: Whether to create destination directories
            
        Returns:
            True if successful
        """
        try:
            if create_dirs:
                dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(dest))
            return True
        except Exception:
            return False
    
    @staticmethod
    def delete_empty_directory(path: Path) -> bool:
        """
        Delete a directory if it's empty.
        
        Returns:
            True if directory was deleted
        """
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def ensure_unique_filename(dest: Path) -> Path:
        """
        Ensure filename is unique by adding suffix if needed.
        
        Args:
            dest: Desired destination path
            
        Returns:
            Unique path (original or with _1, _2, etc.)
        """
        if not dest.exists():
            return dest
        
        counter = 1
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1


def test_file_utils():
    """Quick test of file utilities."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test structure
        (tmpdir / "subdir1" / "subdir1a").mkdir(parents=True)
        (tmpdir / "subdir2").mkdir()
        
        # Create test files
        (tmpdir / "image1.jpg").write_text("fake jpg")
        (tmpdir / "subdir1" / "image2.png").write_text("fake png")
        (tmpdir / "subdir1" / "notes.txt").write_text("notes")
        (tmpdir / "subdir1" / "subdir1a" / "deep_image.tiff").write_text("fake tiff")
        (tmpdir / "subdir2" / "other.xyz").write_text("other file")
        
        # Test classifier
        classifier = FileClassifier()
        assert classifier.classify(Path("test.jpg")) == FileType.IMAGE
        assert classifier.classify(Path("test.pdf")) == FileType.TEXT
        assert classifier.classify(Path("test.xyz")) == FileType.OTHER
        print("✓ Classifier works")
        
        # Test walker
        walker = DirectoryWalker(classifier)
        visited = []
        for dir_path, files in walker.walk_depth_first(tmpdir):
            visited.append((dir_path.relative_to(tmpdir), [f.name for f in files]))
        
        # Should visit deepest first
        print(f"Visited order: {[str(v[0]) for v in visited]}")
        assert len(visited) == 4  # 3 subdirs + root
        
        # Test counts
        counts = walker.count_files_recursive(tmpdir)
        assert counts[FileType.IMAGE] == 3
        assert counts[FileType.TEXT] == 1
        print(f"✓ Counts: {counts}")
        
        # Test file operations
        ops = FileOperations()
        dest = tmpdir / "moved_image.jpg"
        success = ops.copy_file(tmpdir / "image1.jpg", dest)
        assert success and dest.exists()
        print("✓ File operations work")
        
        # Test unique filename
        unique = ops.ensure_unique_filename(dest)
        assert unique.name == "moved_image_1.jpg"
        print("✓ Unique filename works")
        
        # Test delete empty
        empty_dir = tmpdir / "empty"
        empty_dir.mkdir()
        deleted = ops.delete_empty_directory(empty_dir)
        assert deleted and not empty_dir.exists()
        print("✓ Delete empty directory works")
        
        print("\n✓ All file_utils tests passed!")


if __name__ == "__main__":
    test_file_utils()
