"""
Image Hashing Module for the data ordering tool.
Computes perceptual hashes (pHash) for near-duplicate detection
and MD5 hashes for exact duplicate detection.
"""
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from PIL import Image
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    
from .config import IMAGE_EXTENSIONS, IMAGE_WITH_METADATA_SUPPORT


logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """Type of duplicate relationship."""
    EXACT = "exact"           # Identical files (same MD5)
    PERCEPTUAL = "perceptual" # Visually similar (same pHash)
    NEAR = "near"             # Very similar (small pHash difference)


@dataclass
class ImageHash:
    """Hash information for an image."""
    file_path: Path
    md5_hash: str                           # Exact content hash
    phash: Optional[str] = None             # Perceptual hash (None if not an image)
    ahash: Optional[str] = None             # Average hash (alternative)
    file_size: int = 0                      # File size in bytes
    
    def __hash__(self):
        return hash(self.md5_hash)
    
    def __eq__(self, other):
        if isinstance(other, ImageHash):
            return self.md5_hash == other.md5_hash
        return False


@dataclass
class DuplicateGroup:
    """A group of duplicate images."""
    duplicate_type: DuplicateType
    hash_value: str                         # The hash that groups them
    files: List[Path] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.files)
    
    @property
    def primary_file(self) -> Optional[Path]:
        """Get the 'best' file to keep (largest, or first if same size)."""
        if not self.files:
            return None
        # Prefer larger files (higher resolution)
        return max(self.files, key=lambda p: p.stat().st_size if p.exists() else 0)


class ImageHasher:
    """
    Computes and manages image hashes for duplicate detection.
    
    Uses:
    - MD5: For exact duplicate detection (byte-for-byte identical)
    - pHash: Perceptual hash for visually similar images
            (survives resizing, minor edits, format conversion)
    """
    
    # Default threshold for "near" duplicates (hamming distance)
    # pHash typically uses 64-bit hashes, so max distance is 64
    # Lower = more similar required
    NEAR_DUPLICATE_THRESHOLD = 8
    
    def __init__(self, phash_threshold: int = None):
        """
        Initialize the image hasher.
        
        Args:
            phash_threshold: Maximum hamming distance for near-duplicate detection
        """
        self.phash_threshold = phash_threshold or self.NEAR_DUPLICATE_THRESHOLD
        
        if not IMAGEHASH_AVAILABLE:
            logger.warning(
                "imagehash library not available. Install with: pip install imagehash Pillow"
            )
    
    def compute_md5(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Compute MD5 hash of a file.
        
        Args:
            file_path: Path to file
            chunk_size: Read buffer size
            
        Returns:
            Hex string of MD5 hash
        """
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except (IOError, OSError) as e:
            logger.error(f"Error computing MD5 for {file_path}: {e}")
            return ""
    
    def compute_phash(self, file_path: Path) -> Optional[str]:
        """
        Compute perceptual hash (pHash) of an image.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Hex string of pHash, or None if not an image or error
        """
        if not IMAGEHASH_AVAILABLE:
            return None
        
        # Check if it's an image we can process
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Compute perceptual hash
                phash = imagehash.phash(img)
                return str(phash)
                
        except Exception as e:
            logger.warning(f"Error computing pHash for {file_path}: {e}")
            return None
    
    def compute_ahash(self, file_path: Path) -> Optional[str]:
        """
        Compute average hash (aHash) of an image.
        Faster but less accurate than pHash.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Hex string of aHash, or None if not an image or error
        """
        if not IMAGEHASH_AVAILABLE:
            return None
        
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        
        try:
            with Image.open(file_path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                ahash = imagehash.average_hash(img)
                return str(ahash)
        except Exception as e:
            logger.warning(f"Error computing aHash for {file_path}: {e}")
            return None
    
    def hash_image(self, file_path: Path, compute_perceptual: bool = True) -> ImageHash:
        """
        Compute all hashes for an image file.
        
        Args:
            file_path: Path to image file
            compute_perceptual: Whether to compute perceptual hashes
            
        Returns:
            ImageHash with all computed hashes
        """
        md5 = self.compute_md5(file_path)
        
        phash = None
        ahash = None
        if compute_perceptual:
            phash = self.compute_phash(file_path)
            # Only compute ahash if phash succeeded (both require image loading)
            if phash:
                ahash = self.compute_ahash(file_path)
        
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        return ImageHash(
            file_path=file_path,
            md5_hash=md5,
            phash=phash,
            ahash=ahash,
            file_size=file_size
        )
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hex hash strings.
        
        Args:
            hash1: First hash as hex string
            hash2: Second hash as hex string
            
        Returns:
            Number of differing bits
        """
        if not hash1 or not hash2:
            return -1  # Invalid comparison
        
        # Convert hex to integers and XOR
        try:
            h1 = int(hash1, 16)
            h2 = int(hash2, 16)
            xor = h1 ^ h2
            # Count set bits
            return bin(xor).count('1')
        except ValueError:
            return -1
    
    def are_duplicates(self, hash1: ImageHash, hash2: ImageHash) -> Optional[DuplicateType]:
        """
        Check if two images are duplicates.
        
        Args:
            hash1: First image hash
            hash2: Second image hash
            
        Returns:
            DuplicateType if duplicates, None otherwise
        """
        # Exact duplicate (same MD5)
        if hash1.md5_hash and hash2.md5_hash and hash1.md5_hash == hash2.md5_hash:
            return DuplicateType.EXACT
        
        # Perceptual duplicate (same pHash)
        if hash1.phash and hash2.phash:
            if hash1.phash == hash2.phash:
                return DuplicateType.PERCEPTUAL
            
            # Near duplicate (small hamming distance)
            distance = self.hamming_distance(hash1.phash, hash2.phash)
            if 0 < distance <= self.phash_threshold:
                return DuplicateType.NEAR
        
        return None


class HashRegistry:
    """
    Registry for tracking image hashes and finding duplicates.
    """
    
    def __init__(self, hasher: ImageHasher = None):
        """
        Initialize the hash registry.
        
        Args:
            hasher: ImageHasher instance (creates default if None)
        """
        self.hasher = hasher or ImageHasher()
        
        # Storage: file_path -> ImageHash
        self._hashes: Dict[Path, ImageHash] = {}
        
        # Index: md5 -> set of file paths
        self._md5_index: Dict[str, Set[Path]] = {}
        
        # Index: phash -> set of file paths
        self._phash_index: Dict[str, Set[Path]] = {}
    
    def add(self, file_path: Path, compute_perceptual: bool = True) -> ImageHash:
        """
        Add a file to the registry.
        
        Args:
            file_path: Path to image file
            compute_perceptual: Whether to compute perceptual hashes
            
        Returns:
            The computed ImageHash
        """
        # Compute hashes
        img_hash = self.hasher.hash_image(file_path, compute_perceptual)
        
        # Store in main dict
        self._hashes[file_path] = img_hash
        
        # Update MD5 index
        if img_hash.md5_hash:
            if img_hash.md5_hash not in self._md5_index:
                self._md5_index[img_hash.md5_hash] = set()
            self._md5_index[img_hash.md5_hash].add(file_path)
        
        # Update pHash index
        if img_hash.phash:
            if img_hash.phash not in self._phash_index:
                self._phash_index[img_hash.phash] = set()
            self._phash_index[img_hash.phash].add(file_path)
        
        return img_hash
    
    def get(self, file_path: Path) -> Optional[ImageHash]:
        """Get hash for a file."""
        return self._hashes.get(file_path)
    
    def find_exact_duplicates(self, file_path: Path) -> List[Path]:
        """
        Find exact duplicates of a file.
        
        Args:
            file_path: Path to check
            
        Returns:
            List of duplicate file paths (excluding the input file)
        """
        img_hash = self._hashes.get(file_path)
        if not img_hash or not img_hash.md5_hash:
            return []
        
        duplicates = self._md5_index.get(img_hash.md5_hash, set())
        return [p for p in duplicates if p != file_path]
    
    def find_perceptual_duplicates(self, file_path: Path) -> List[Path]:
        """
        Find perceptually identical images (same pHash).
        
        Args:
            file_path: Path to check
            
        Returns:
            List of duplicate file paths (excluding the input file)
        """
        img_hash = self._hashes.get(file_path)
        if not img_hash or not img_hash.phash:
            return []
        
        duplicates = self._phash_index.get(img_hash.phash, set())
        return [p for p in duplicates if p != file_path]
    
    def find_near_duplicates(self, file_path: Path, threshold: int = None) -> List[Tuple[Path, int]]:
        """
        Find near-duplicate images (similar pHash).
        
        Args:
            file_path: Path to check
            threshold: Maximum hamming distance (uses hasher default if None)
            
        Returns:
            List of (file_path, distance) tuples, sorted by distance
        """
        img_hash = self._hashes.get(file_path)
        if not img_hash or not img_hash.phash:
            return []
        
        threshold = threshold or self.hasher.phash_threshold
        results = []
        
        for other_path, other_hash in self._hashes.items():
            if other_path == file_path or not other_hash.phash:
                continue
            
            distance = self.hasher.hamming_distance(img_hash.phash, other_hash.phash)
            if 0 < distance <= threshold:
                results.append((other_path, distance))
        
        return sorted(results, key=lambda x: x[1])
    
    def get_all_duplicate_groups(self) -> Dict[DuplicateType, List[DuplicateGroup]]:
        """
        Get all duplicate groups in the registry.
        
        Returns:
            Dict mapping DuplicateType to list of DuplicateGroups
        """
        result = {
            DuplicateType.EXACT: [],
            DuplicateType.PERCEPTUAL: [],
        }
        
        # Exact duplicates (by MD5)
        for md5, paths in self._md5_index.items():
            if len(paths) > 1:
                result[DuplicateType.EXACT].append(DuplicateGroup(
                    duplicate_type=DuplicateType.EXACT,
                    hash_value=md5,
                    files=list(paths)
                ))
        
        # Perceptual duplicates (by pHash)
        for phash, paths in self._phash_index.items():
            if len(paths) > 1:
                # Exclude if already in exact duplicates
                non_exact = []
                for p in paths:
                    h = self._hashes.get(p)
                    if h and len(self._md5_index.get(h.md5_hash, set())) <= 1:
                        non_exact.append(p)
                
                if len(non_exact) > 1:
                    result[DuplicateType.PERCEPTUAL].append(DuplicateGroup(
                        duplicate_type=DuplicateType.PERCEPTUAL,
                        hash_value=phash,
                        files=non_exact
                    ))
        
        return result
    
    @property
    def total_files(self) -> int:
        """Total number of files in registry."""
        return len(self._hashes)
    
    @property
    def exact_duplicate_count(self) -> int:
        """Number of files that are exact duplicates."""
        return sum(len(paths) - 1 for paths in self._md5_index.values() if len(paths) > 1)
    
    @property
    def perceptual_duplicate_count(self) -> int:
        """Number of files that are perceptual duplicates."""
        return sum(len(paths) - 1 for paths in self._phash_index.values() if len(paths) > 1)
    
    def to_dict(self) -> Dict[str, dict]:
        """
        Export registry to dict (for serialization).
        
        Returns:
            Dict with file paths as keys and hash info as values
        """
        return {
            str(path): {
                'md5': h.md5_hash,
                'phash': h.phash,
                'ahash': h.ahash,
                'size': h.file_size,
            }
            for path, h in self._hashes.items()
        }
    
    def from_dict(self, data: Dict[str, dict]):
        """
        Import registry from dict.
        
        Args:
            data: Dict with file paths as keys and hash info as values
        """
        for path_str, info in data.items():
            path = Path(path_str)
            img_hash = ImageHash(
                file_path=path,
                md5_hash=info.get('md5', ''),
                phash=info.get('phash'),
                ahash=info.get('ahash'),
                file_size=info.get('size', 0)
            )
            self._hashes[path] = img_hash
            
            if img_hash.md5_hash:
                if img_hash.md5_hash not in self._md5_index:
                    self._md5_index[img_hash.md5_hash] = set()
                self._md5_index[img_hash.md5_hash].add(path)
            
            if img_hash.phash:
                if img_hash.phash not in self._phash_index:
                    self._phash_index[img_hash.phash] = set()
                self._phash_index[img_hash.phash].add(path)


# Convenience functions
def hash_image(file_path: Path | str) -> ImageHash:
    """Quick function to hash a single image."""
    hasher = ImageHasher()
    return hasher.hash_image(Path(file_path))


def find_duplicates_in_folder(folder: Path | str, recursive: bool = True) -> Dict[DuplicateType, List[DuplicateGroup]]:
    """
    Find all duplicates in a folder.
    
    Args:
        folder: Folder to scan
        recursive: Whether to scan subdirectories
        
    Returns:
        Dict mapping DuplicateType to list of DuplicateGroups
    """
    folder = Path(folder)
    registry = HashRegistry()
    
    pattern = '**/*' if recursive else '*'
    for file_path in folder.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            logger.info(f"Hashing: {file_path}")
            registry.add(file_path)
    
    return registry.get_all_duplicate_groups()
