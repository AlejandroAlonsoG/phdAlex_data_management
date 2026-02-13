"""
File Scanner Module for the data ordering tool.
Traverses source directories, classifies files, extracts patterns,
computes hashes, and builds a comprehensive file registry.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Generator, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from .config import (
    Config, config, IMAGE_EXTENSIONS, TEXT_EXTENSIONS,
    COLLECTIONS, detect_collection, CollectionConfig
)
from .file_utils import FileClassifier, FileType, FileInfo, DirectoryWalker
from .pattern_extractor import PatternExtractor, ExtractionResult
from .image_hasher import ImageHasher, HashRegistry, ImageHash, DuplicateType


logger = logging.getLogger(__name__)


class ScanStatus(Enum):
    """Status of file scan."""
    PENDING = "pending"
    SCANNED = "scanned"
    HASHED = "hashed"
    PATTERN_EXTRACTED = "pattern_extracted"
    COLLECTION_ASSIGNED = "collection_assigned"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ScannedFile:
    """Complete information about a scanned file."""
    # Basic info
    path: Path
    file_type: FileType
    file_size: int
    modified_time: datetime
    
    # Pattern extraction results
    extraction: Optional[ExtractionResult] = None
    specimen_id: Optional[str] = None
    collection: Optional[str] = None  # Collection code (LH, BUE, etc.)
    
    # Hash info
    md5_hash: Optional[str] = None
    phash: Optional[str] = None
    
    # Duplicate tracking
    is_duplicate: bool = False
    duplicate_of: Optional[Path] = None
    duplicate_type: Optional[DuplicateType] = None
    
    # Processing status
    status: ScanStatus = ScanStatus.PENDING
    error_message: Optional[str] = None
    
    # LLM processing (to be filled later)
    needs_llm_review: bool = False
    llm_result: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'path': str(self.path),
            'file_type': self.file_type.value,
            'file_size': self.file_size,
            'modified_time': self.modified_time.isoformat(),
            'specimen_id': self.specimen_id,
            'collection': self.collection,
            'md5_hash': self.md5_hash,
            'phash': self.phash,
            'is_duplicate': self.is_duplicate,
            'duplicate_of': str(self.duplicate_of) if self.duplicate_of else None,
            'duplicate_type': self.duplicate_type.value if self.duplicate_type else None,
            'status': self.status.value,
            'error_message': self.error_message,
            'needs_llm_review': self.needs_llm_review,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ScannedFile':
        """Create from dictionary."""
        return cls(
            path=Path(data['path']),
            file_type=FileType(data['file_type']),
            file_size=data['file_size'],
            modified_time=datetime.fromisoformat(data['modified_time']),
            specimen_id=data.get('specimen_id'),
            collection=data.get('collection'),
            md5_hash=data.get('md5_hash'),
            phash=data.get('phash'),
            is_duplicate=data.get('is_duplicate', False),
            duplicate_of=Path(data['duplicate_of']) if data.get('duplicate_of') else None,
            duplicate_type=DuplicateType(data['duplicate_type']) if data.get('duplicate_type') else None,
            status=ScanStatus(data.get('status', 'pending')),
            error_message=data.get('error_message'),
            needs_llm_review=data.get('needs_llm_review', False),
        )


@dataclass
class ScanProgress:
    """Tracks scanning progress."""
    total_files: int = 0
    scanned_files: int = 0
    hashed_files: int = 0
    pattern_extracted: int = 0
    errors: int = 0
    
    # By file type
    images: int = 0
    text_files: int = 0
    other_files: int = 0
    
    # Duplicates found
    exact_duplicates: int = 0
    perceptual_duplicates: int = 0
    
    # Collections
    by_collection: Dict[str, int] = field(default_factory=dict)
    
    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.scanned_files / self.total_files) * 100
    
    def __str__(self) -> str:
        return (
            f"Progress: {self.scanned_files}/{self.total_files} ({self.progress_percent:.1f}%) | "
            f"Images: {self.images} | Text: {self.text_files} | Other: {self.other_files} | "
            f"Duplicates: {self.exact_duplicates + self.perceptual_duplicates} | Errors: {self.errors}"
        )


class FileScanner:
    """
    Scans directories and builds a comprehensive file registry.
    
    Workflow:
    1. Count total files (for progress)
    2. Scan each file:
       - Classify by type
       - Extract patterns (specimen ID, dates, etc.)
       - Compute hashes (MD5 + pHash for images)
       - Detect collection from prefix/path
    3. Find duplicates
    4. Mark files needing LLM review
    
    NOTE: In the production pipeline (PipelineOrchestrator), this scanner is 
    typically instantiated with hash_images=False and extract_patterns=False
    to avoid redundant work. The orchestrator has dedicated pipeline stages
    for pattern extraction (_run_pattern_extraction), hashing (_run_hashing),
    and deduplication (_run_deduplication) that consume and store results 
    in the pipeline state. 
    
    However, the hashing/pattern/deduplication code is intentionally left here
    because:
    - It provides a complete, self-contained scanning utility for standalone use
    - It enables quick iteration during development/testing
    - The logic is reusable for batch operations or alternative workflows
    
    When using FileScanner standalone (not via PipelineOrchestrator), you can 
    enable hash_images=True and extract_patterns=True to get a fully 
    enriched file registry in one pass.
    """
    
    def __init__(
        self,
        config: Config = None,
        hash_images: bool = True,
        extract_patterns: bool = True,
        progress_callback: Callable[[ScanProgress], None] = None
    ):
        """
        Initialize the file scanner.
        
        Args:
            config: Configuration object
            hash_images: Whether to compute image hashes
            extract_patterns: Whether to extract patterns from paths
            progress_callback: Function called with progress updates
        """
        self.config = config or globals()['config']
        self.hash_images = hash_images
        self.extract_patterns = extract_patterns
        self.progress_callback = progress_callback
        
        # Components
        self.classifier = FileClassifier()
        self.pattern_extractor = PatternExtractor()
        self.hasher = ImageHasher() if hash_images else None
        self.hash_registry = HashRegistry() if hash_images else None
        
        # Results storage
        self.files: Dict[Path, ScannedFile] = {}
        self.progress = ScanProgress()
        
        # Tracking
        self._scanned_dirs: Set[Path] = set()
    
    def count_files(self, directories: List[Path]) -> int:
        """
        Count total files in directories (for progress tracking).
        
        Args:
            directories: List of directories to scan
            
        Returns:
            Total file count
        """
        count = 0
        for directory in directories:
            if directory.exists():
                for _ in directory.rglob('*'):
                    if _.is_file():
                        count += 1
        return count
    
    def scan_directories(
        self,
        directories: List[Path],
        skip_hidden: bool = True
    ) -> Generator[ScannedFile, None, None]:
        """
        Scan multiple directories and yield scanned files.
        
        Args:
            directories: List of directories to scan
            skip_hidden: Whether to skip hidden files/folders
            
        Yields:
            ScannedFile for each file found
        """
        # Count total first
        logger.info(f"Counting files in {len(directories)} directories...")
        self.progress.total_files = self.count_files(directories)
        logger.info(f"Found {self.progress.total_files} files to scan")
        
        for directory in directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue
            
            logger.info(f"Scanning: {directory}")
            self._scanned_dirs.add(directory)
            
            for scanned_file in self._scan_directory(directory, skip_hidden):
                yield scanned_file
        
        # After scanning, find duplicates
        if self.hash_images and self.hash_registry:
            self._find_duplicates()
        
        logger.info(f"Scan complete: {self.progress}")
    
    def _scan_directory(
        self,
        directory: Path,
        skip_hidden: bool
    ) -> Generator[ScannedFile, None, None]:
        """Scan a single directory."""
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip hidden files
            if skip_hidden and any(part.startswith('.') for part in file_path.parts):
                continue
            
            try:
                scanned = self._scan_file(file_path)
                self.files[file_path] = scanned
                self.progress.scanned_files += 1
                
                # Update progress callback
                if self.progress_callback and self.progress.scanned_files % 100 == 0:
                    self.progress_callback(self.progress)
                
                yield scanned
                
            except Exception as e:
                logger.error(f"Error scanning {file_path}: {e}")
                self.progress.errors += 1
                
                # Create error entry
                error_file = ScannedFile(
                    path=file_path,
                    file_type=FileType.OTHER,
                    file_size=0,
                    modified_time=datetime.now(),
                    status=ScanStatus.ERROR,
                    error_message=str(e)
                )
                self.files[file_path] = error_file
                yield error_file
    
    def _scan_file(self, file_path: Path) -> ScannedFile:
        """
        Scan a single file and extract all information.
        
        Args:
            file_path: Path to file
            
        Returns:
            ScannedFile with all extracted information
        """
        # Basic file info
        stat = file_path.stat()
        file_type = self.classifier.classify(file_path)
        
        scanned = ScannedFile(
            path=file_path,
            file_type=file_type,
            file_size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            status=ScanStatus.SCANNED
        )
        
        # Update type counts
        if file_type == FileType.IMAGE:
            self.progress.images += 1
        elif file_type == FileType.TEXT:
            self.progress.text_files += 1
        else:
            self.progress.other_files += 1
        
        # Extract patterns
        if self.extract_patterns:
            self._extract_patterns(scanned)
        
        # Compute hashes for images
        if self.hash_images and file_type == FileType.IMAGE:
            self._compute_hashes(scanned)
        
        # Collection detection and LLM-review flagging depend on pattern
        # extraction results and are only meaningful in standalone mode.
        # In the pipeline, the orchestrator handles these in dedicated stages
        # (_run_pattern_extraction, _run_hashing, _run_deduplication).
        if self.extract_patterns:
            self._detect_collection(scanned)
            self._check_needs_llm_review(scanned)
        
        scanned.status = ScanStatus.COMPLETE
        return scanned
    
    def _extract_patterns(self, scanned: ScannedFile):
        """Extract patterns from file path."""
        try:
            extraction = self.pattern_extractor.extract(scanned.path)
            scanned.extraction = extraction
            
            if extraction.specimen_id:
                scanned.specimen_id = extraction.specimen_id.specimen_id
            
            scanned.status = ScanStatus.PATTERN_EXTRACTED
            self.progress.pattern_extracted += 1
            
        except Exception as e:
            logger.warning(f"Pattern extraction failed for {scanned.path}: {e}")
    
    def _compute_hashes(self, scanned: ScannedFile):
        """Compute hashes for image file."""
        try:
            img_hash = self.hash_registry.add(scanned.path)
            scanned.md5_hash = img_hash.md5_hash
            scanned.phash = img_hash.phash
            scanned.status = ScanStatus.HASHED
            self.progress.hashed_files += 1
            
        except Exception as e:
            logger.warning(f"Hashing failed for {scanned.path}: {e}")
    
    def _detect_collection(self, scanned: ScannedFile):
        """Detect which collection the file belongs to."""
        # Try by prefix first (most reliable)
        prefix = None
        if scanned.extraction and scanned.extraction.specimen_id:
            prefix = scanned.extraction.specimen_id.prefix
        
        collection = detect_collection(path=scanned.path, prefix=prefix)
        
        if collection:
            scanned.collection = collection.code
            scanned.status = ScanStatus.COLLECTION_ASSIGNED
            
            # Update collection counts
            if collection.code not in self.progress.by_collection:
                self.progress.by_collection[collection.code] = 0
            self.progress.by_collection[collection.code] += 1
    
    def _check_needs_llm_review(self, scanned: ScannedFile):
        """Determine if file needs LLM review for classification."""
        # Needs review if:
        # 1. Is an image but no specimen ID found
        # 2. No collection could be determined
        # 3. Has potential taxonomy info in path (for validation)
        
        if scanned.file_type == FileType.IMAGE:
            if not scanned.specimen_id:
                scanned.needs_llm_review = True
            elif not scanned.collection:
                scanned.needs_llm_review = True
    
    def _find_duplicates(self):
        """Find and mark duplicate files after scanning."""
        if not self.hash_registry:
            return
        
        logger.info("Finding duplicates...")
        
        duplicate_groups = self.hash_registry.get_all_duplicate_groups()
        
        # Process exact duplicates
        for group in duplicate_groups[DuplicateType.EXACT]:
            primary = group.primary_file
            for dup_path in group.files:
                if dup_path != primary and dup_path in self.files:
                    self.files[dup_path].is_duplicate = True
                    self.files[dup_path].duplicate_of = primary
                    self.files[dup_path].duplicate_type = DuplicateType.EXACT
                    self.progress.exact_duplicates += 1
        
        # Process perceptual duplicates
        for group in duplicate_groups[DuplicateType.PERCEPTUAL]:
            primary = group.primary_file
            for dup_path in group.files:
                if dup_path != primary and dup_path in self.files:
                    # Only mark as perceptual dup if not already exact
                    if not self.files[dup_path].is_duplicate:
                        self.files[dup_path].is_duplicate = True
                        self.files[dup_path].duplicate_of = primary
                        self.files[dup_path].duplicate_type = DuplicateType.PERCEPTUAL
                        self.progress.perceptual_duplicates += 1
        
        logger.info(
            f"Found {self.progress.exact_duplicates} exact duplicates, "
            f"{self.progress.perceptual_duplicates} perceptual duplicates"
        )
    
    def get_files_by_type(self, file_type: FileType) -> List[ScannedFile]:
        """Get all files of a specific type."""
        return [f for f in self.files.values() if f.file_type == file_type]
    
    def get_files_by_collection(self, collection_code: str) -> List[ScannedFile]:
        """Get all files for a specific collection."""
        return [f for f in self.files.values() if f.collection == collection_code]
    
    def get_files_needing_review(self) -> List[ScannedFile]:
        """Get files that need LLM review."""
        return [f for f in self.files.values() if f.needs_llm_review]
    
    def get_duplicates(self) -> List[ScannedFile]:
        """Get all duplicate files."""
        return [f for f in self.files.values() if f.is_duplicate]
    
    def get_unique_files(self) -> List[ScannedFile]:
        """Get all non-duplicate files."""
        return [f for f in self.files.values() if not f.is_duplicate]
    
    def save_state(self, file_path: Path):
        """
        Save scan state to JSON file (for resume capability).
        
        Args:
            file_path: Path to save state
        """
        state = {
            'scanned_dirs': [str(d) for d in self._scanned_dirs],
            'progress': {
                'total_files': self.progress.total_files,
                'scanned_files': self.progress.scanned_files,
                'images': self.progress.images,
                'text_files': self.progress.text_files,
                'other_files': self.progress.other_files,
                'exact_duplicates': self.progress.exact_duplicates,
                'perceptual_duplicates': self.progress.perceptual_duplicates,
                'errors': self.progress.errors,
                'by_collection': self.progress.by_collection,
            },
            'files': {str(p): f.to_dict() for p, f in self.files.items()},
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved scan state to {file_path}")
    
    def load_state(self, file_path: Path) -> bool:
        """
        Load scan state from JSON file.
        
        Args:
            file_path: Path to load state from
            
        Returns:
            True if loaded successfully
        """
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self._scanned_dirs = {Path(d) for d in state.get('scanned_dirs', [])}
            
            prog = state.get('progress', {})
            self.progress.total_files = prog.get('total_files', 0)
            self.progress.scanned_files = prog.get('scanned_files', 0)
            self.progress.images = prog.get('images', 0)
            self.progress.text_files = prog.get('text_files', 0)
            self.progress.other_files = prog.get('other_files', 0)
            self.progress.exact_duplicates = prog.get('exact_duplicates', 0)
            self.progress.perceptual_duplicates = prog.get('perceptual_duplicates', 0)
            self.progress.errors = prog.get('errors', 0)
            self.progress.by_collection = prog.get('by_collection', {})
            
            self.files = {
                Path(p): ScannedFile.from_dict(data)
                for p, data in state.get('files', {}).items()
            }
            
            logger.info(f"Loaded scan state: {self.progress}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state from {file_path}: {e}")
            return False
    
    def get_summary(self) -> dict:
        """Get a summary of the scan results."""
        return {
            'total_files': self.progress.total_files,
            'images': self.progress.images,
            'text_files': self.progress.text_files,
            'other_files': self.progress.other_files,
            'with_specimen_id': sum(1 for f in self.files.values() if f.specimen_id),
            'without_specimen_id': sum(1 for f in self.files.values() if not f.specimen_id and f.file_type == FileType.IMAGE),
            'exact_duplicates': self.progress.exact_duplicates,
            'perceptual_duplicates': self.progress.perceptual_duplicates,
            'needs_llm_review': sum(1 for f in self.files.values() if f.needs_llm_review),
            'by_collection': self.progress.by_collection,
            'errors': self.progress.errors,
        }


# Convenience function
def scan_directory(
    directory: Path | str,
    hash_images: bool = True,
    progress_callback: Callable[[ScanProgress], None] = None
) -> FileScanner:
    """
    Quick function to scan a directory.
    
    Args:
        directory: Directory to scan
        hash_images: Whether to compute image hashes
        progress_callback: Progress update callback
        
    Returns:
        FileScanner with results
    """
    scanner = FileScanner(
        hash_images=hash_images,
        progress_callback=progress_callback
    )
    
    # Consume the generator
    for _ in scanner.scan_directories([Path(directory)]):
        pass
    
    return scanner
