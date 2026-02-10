"""
Main Orchestrator Module for the data ordering tool.
Coordinates all modules to execute the full pipeline:
scan → extract patterns → analyze with LLM → hash → deduplicate → organize
"""
import os
import json
import logging
import shutil
import uuid as uuid_lib
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import config, detect_collection, COLLECTIONS, get_macroclass_folder
from .logger_module import DataOrderingLogger, LogAction
from .excel_manager import ExcelManager, ImageRecord, TextFileRecord, OtherFileRecord
from .file_utils import DirectoryWalker, FileClassifier, FileType, FileInfo
from .pattern_extractor import PatternExtractor, ExtractionResult
from .image_hasher import ImageHasher, HashRegistry, DuplicateGroup
from .file_scanner import FileScanner, ScannedFile, ScanProgress
from .llm_integration import (
    GeminiClient, DirectoryAnalyzer, DirectoryAnalysis, 
    MockGeminiClient, GENAI_AVAILABLE, OPENAI_AVAILABLE, get_llm_client
)
from .interaction_manager import (
    InteractionManager, InteractionMode, DecisionType, DecisionOutcome,
    DecisionRequest, DecisionResult,
    create_numeric_id_decision, create_metadata_conflict_decision,
    create_duplicate_decision, create_unknown_collection_decision
)


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the processing pipeline."""
    INIT = "init"
    SCANNING = "scanning"
    LLM_ANALYSIS = "llm_analysis"
    PATTERN_EXTRACTION = "pattern_extraction"
    HASHING = "hashing"
    DEDUPLICATION = "deduplication"
    REGISTRY_GENERATION = "registry_generation"  # Log to Excel FIRST
    ORGANIZING = "organizing"                     # Then move files
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessedFile:
    """A file that has been fully processed through the pipeline."""
    # Original file info
    original_path: Path
    filename: str
    file_size: int
    file_type: FileType
    
    # Generated UUID for this file
    file_uuid: Optional[str] = None
    
    # Extracted metadata
    specimen_id: Optional[str] = None
    collection_code: Optional[str] = None  # LH, BUE, MON
    macroclass: Optional[str] = None       # Arthropoda, Pisces, etc.
    taxonomic_class: Optional[str] = None  # Insecta, Osteichthyes, etc.
    determination: Optional[str] = None    # Genus level
    campaign_year: Optional[int] = None
    fecha_captura: Optional[str] = None    # Date from EXIF DateTimeOriginal (YYYYMMDD)
    
    # Hashing info
    md5_hash: Optional[str] = None
    phash: Optional[str] = None
    
    # Duplicate info
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None  # Path to original
    
    # Decision tracking
    needs_review: bool = False
    review_reason: Optional[str] = None
    
    # Destination
    destination_path: Optional[Path] = None
    new_filename: Optional[str] = None  # Renamed filename
    moved: bool = False
    
    # Errors
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'original_path': str(self.original_path),
            'filename': self.filename,
            'file_size': self.file_size,
            'file_type': self.file_type.value if self.file_type else None,
            'file_uuid': self.file_uuid,
            'specimen_id': self.specimen_id,
            'collection_code': self.collection_code,
            'macroclass': self.macroclass,
            'taxonomic_class': self.taxonomic_class,
            'determination': self.determination,
            'campaign_year': self.campaign_year,
            'fecha_captura': self.fecha_captura,
            'md5_hash': self.md5_hash,
            'phash': self.phash,
            'is_duplicate': self.is_duplicate,
            'duplicate_of': str(self.duplicate_of) if self.duplicate_of else None,
            'needs_review': self.needs_review,
            'review_reason': self.review_reason,
            'destination_path': str(self.destination_path) if self.destination_path else None,
            'new_filename': self.new_filename,
            'moved': self.moved,
            'error': self.error,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessedFile':
        return cls(
            original_path=Path(data['original_path']),
            filename=data['filename'],
            file_size=data['file_size'],
            file_type=FileType(data['file_type']) if data.get('file_type') else None,
            file_uuid=data.get('file_uuid'),
            specimen_id=data.get('specimen_id'),
            collection_code=data.get('collection_code'),
            macroclass=data.get('macroclass'),
            taxonomic_class=data.get('taxonomic_class'),
            determination=data.get('determination'),
            campaign_year=data.get('campaign_year'),
            fecha_captura=data.get('fecha_captura'),
            md5_hash=data.get('md5_hash'),
            phash=data.get('phash'),
            is_duplicate=data.get('is_duplicate', False),
            duplicate_of=data.get('duplicate_of'),
            needs_review=data.get('needs_review', False),
            review_reason=data.get('review_reason'),
            destination_path=Path(data['destination_path']) if data.get('destination_path') else None,
            new_filename=data.get('new_filename'),
            moved=data.get('moved', False),
            error=data.get('error'),
        )


@dataclass
class PipelineState:
    """State of the pipeline for resume capability."""
    stage: PipelineStage = PipelineStage.INIT
    source_directories: List[str] = field(default_factory=list)
    output_directory: Optional[str] = None
    
    # Progress tracking
    total_files: int = 0
    files_scanned: int = 0
    directories_analyzed: int = 0
    files_hashed: int = 0
    files_organized: int = 0
    
    # Results
    processed_files: Dict[str, dict] = field(default_factory=dict)  # path -> ProcessedFile.to_dict()
    directory_analyses: Dict[str, dict] = field(default_factory=dict)  # dir_path -> DirectoryAnalysis.to_dict()
    duplicate_groups: List[dict] = field(default_factory=list)
    
    # Errors
    errors: List[dict] = field(default_factory=list)
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def save(self, path: Path):
        """Save state to file."""
        data = {
            'stage': self.stage.value,
            'source_directories': self.source_directories,
            'output_directory': self.output_directory,
            'total_files': self.total_files,
            'files_scanned': self.files_scanned,
            'directories_analyzed': self.directories_analyzed,
            'files_hashed': self.files_hashed,
            'files_organized': self.files_organized,
            'processed_files': self.processed_files,
            'directory_analyses': self.directory_analyses,
            'duplicate_groups': self.duplicate_groups,
            'errors': self.errors,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        """Load state from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = cls()
        state.stage = PipelineStage(data['stage'])
        state.source_directories = data['source_directories']
        state.output_directory = data.get('output_directory')
        state.total_files = data['total_files']
        state.files_scanned = data['files_scanned']
        state.directories_analyzed = data['directories_analyzed']
        state.files_hashed = data['files_hashed']
        state.files_organized = data['files_organized']
        state.processed_files = data['processed_files']
        state.directory_analyses = data['directory_analyses']
        state.duplicate_groups = data.get('duplicate_groups', [])
        state.errors = data.get('errors', [])
        state.started_at = data.get('started_at')
        state.completed_at = data.get('completed_at')
        
        return state


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates the full processing pipeline.
    
    Pipeline stages:
    1. SCANNING: Walk directories and discover all files
    2. LLM_ANALYSIS: Analyze directories with LLM (one call per directory)
    3. PATTERN_EXTRACTION: Extract specimen IDs using LLM-provided regex
    4. HASHING: Compute MD5 and perceptual hashes for images
    5. DEDUPLICATION: Identify and group duplicates
    6. ORGANIZING: Move files to collection-based folders (with renaming)
    7. REGISTRY_GENERATION: Create Excel registries
    
    File handling:
    - Images: Organized into Collection/Campaign/Macroclass/Class/Determination folders
    - Text files: Moved to Annotations folder, renamed with UUID prefix
    - Other files: Moved to Other_Files folder, renamed with UUID prefix
    """
    
    def __init__(
        self,
        source_dirs: List[Path],
        output_dir: Path = None,
        output_base: Path = None,  # Alias for output_dir
        state_path: Path = None,
        use_llm: bool = True,
        dry_run: bool = False,
        interactive: bool = False,  # Legacy parameter
        interaction_mode: str = None,  # New: 'interactive', 'deferred', 'auto_accept', 'step_by_step'
        progress_callback: Callable[[str, int, int], None] = None,
        phash_threshold: int = 8,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            source_dirs: List of directories to process
            output_dir: Output directory for organized files
            output_base: Alias for output_dir
            state_path: Path to save/load state for resume
            use_llm: Whether to use LLM for directory analysis
            dry_run: If True, don't actually move files
            interactive: Legacy param - if True, use interactive mode
            interaction_mode: 'interactive', 'deferred', 'auto_accept', or 'step_by_step'
            progress_callback: Called with (stage, current, total)
            phash_threshold: Threshold for perceptual hash matching
        """
        self.source_dirs = [Path(d) for d in source_dirs]
        self.output_dir = Path(output_base or output_dir) if (output_base or output_dir) else config.output_base_dir
        self.output_base = self.output_dir  # Alias for compatibility
        self.state_path = state_path or (self.output_dir / "pipeline_state.json")
        self.use_llm = use_llm
        self.dry_run = dry_run
        self.progress_callback = progress_callback
        self.phash_threshold = phash_threshold
        
        # Determine interaction mode
        if interaction_mode:
            mode_map = {
                'interactive': InteractionMode.INTERACTIVE,
                'deferred': InteractionMode.DEFERRED,
                'auto_accept': InteractionMode.AUTO_ACCEPT,
                'step_by_step': InteractionMode.STEP_BY_STEP,
            }
            self._interaction_mode = mode_map.get(interaction_mode, InteractionMode.DEFERRED)
        elif interactive:
            self._interaction_mode = InteractionMode.INTERACTIVE
        else:
            self._interaction_mode = InteractionMode.DEFERRED
        
        # Initialize components
        self.file_scanner = FileScanner()
        self.pattern_extractor = PatternExtractor()
        self.image_hasher = ImageHasher()
        self.hash_registry = HashRegistry()
        
        # Interaction manager for decisions
        review_dir = self.output_dir / "Manual_Review"
        self.interaction_manager = InteractionManager(
            mode=self._interaction_mode,
            review_base_dir=review_dir,
            log_decisions=True,
        )
        
        # LLM components (initialized lazily)
        self._directory_analyzer: Optional[DirectoryAnalyzer] = None
        
        # State
        self.state = PipelineState()
        
        # Logger
        self.action_logger = DataOrderingLogger(
            log_dir=self.output_dir / "logs"
        )
    
    @property
    def directory_analyzer(self) -> DirectoryAnalyzer:
        """Lazy initialization of directory analyzer based on config.llm_provider."""
        if self._directory_analyzer is None:
            if self.use_llm:
                # Check if the configured provider's dependencies are available
                provider = config.llm_provider
                can_use_llm = (
                    (provider == 'github' and OPENAI_AVAILABLE) or
                    (provider == 'gemini' and GENAI_AVAILABLE) or
                    (provider not in ('github', 'gemini') and (GENAI_AVAILABLE or OPENAI_AVAILABLE))
                )
                
                if can_use_llm:
                    try:
                        client = get_llm_client(provider)
                        self._directory_analyzer = DirectoryAnalyzer(
                            client=client,
                            cache_path=self.output_dir / "llm_cache.json"
                        )
                        logger.info(f"Using LLM provider: {provider}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {provider} LLM client: {e}")
                        logger.info("Falling back to mock client")
                        self._directory_analyzer = DirectoryAnalyzer(
                            client=MockGeminiClient(),
                            cache_path=self.output_dir / "llm_cache.json"
                        )
                else:
                    logger.warning(f"LLM provider '{provider}' not available (missing dependencies)")
                    self._directory_analyzer = DirectoryAnalyzer(
                        client=MockGeminiClient(),
                        cache_path=self.output_dir / "llm_cache.json"
                    )
            else:
                self._directory_analyzer = DirectoryAnalyzer(
                    client=MockGeminiClient(),
                    cache_path=self.output_dir / "llm_cache.json"
                )
        return self._directory_analyzer
    
    def _report_progress(self, stage: str, current: int, total: int):
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(stage, current, total)
    
    def _save_state(self):
        """Save current state."""
        self.state.save(self.state_path)
    
    def run(self, resume: bool = False) -> PipelineState:
        """
        Run the full pipeline.
        
        Args:
            resume: If True, try to resume from saved state
            
        Returns:
            Final PipelineState
        """
        # Try to resume
        if resume and self.state_path.exists():
            try:
                self.state = PipelineState.load(self.state_path)
                logger.info(f"Resuming from stage: {self.state.stage.value}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}. Starting fresh.")
                self.state = PipelineState()
        
        # Initialize
        self.state.source_directories = [str(d) for d in self.source_dirs]
        self.state.output_directory = str(self.output_dir)
        
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()
        
        try:
            # Run stages with optional confirmation points
            if self.state.stage in [PipelineStage.INIT, PipelineStage.SCANNING]:
                self._run_scanning()
                if not self._confirm_stage_completion("SCANNING"):
                    return self.state
            
            if self.state.stage == PipelineStage.LLM_ANALYSIS:
                self._run_llm_analysis()
                if not self._confirm_stage_completion("LLM_ANALYSIS"):
                    return self.state
            
            if self.state.stage == PipelineStage.PATTERN_EXTRACTION:
                self._run_pattern_extraction()
                if not self._confirm_stage_completion("PATTERN_EXTRACTION"):
                    return self.state
            
            if self.state.stage == PipelineStage.HASHING:
                self._run_hashing()
                if not self._confirm_stage_completion("HASHING"):
                    return self.state
            
            if self.state.stage == PipelineStage.DEDUPLICATION:
                self._run_deduplication()
                if not self._confirm_stage_completion("DEDUPLICATION"):
                    return self.state
            
            # IMPORTANT: Log to Excel FIRST, then move files
            if self.state.stage == PipelineStage.REGISTRY_GENERATION:
                self._run_registry_generation()
                if not self._confirm_stage_completion("REGISTRY_GENERATION"):
                    return self.state
            
            if self.state.stage == PipelineStage.ORGANIZING:
                self._run_organizing()
                # No confirmation needed after final stage
            
            self.state.stage = PipelineStage.COMPLETED
            self.state.completed_at = datetime.now().isoformat()
            self._save_state()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.state.stage = PipelineStage.FAILED
            self.state.errors.append({
                'stage': self.state.stage.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            self._save_state()
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return self.state
    
    def _confirm_stage_completion(self, stage_name: str) -> bool:
        """
        Confirm stage completion in step-by-step mode.
        
        Args:
            stage_name: Name of the completed stage
            
        Returns:
            True to continue, False to abort
        """
        if not self.interaction_manager.is_step_by_step():
            return True  # Auto-continue in other modes
        
        # Build summary data based on stage
        summary_data = {}
        sample_items = []
        description = ""
        
        if stage_name == "SCANNING":
            description = "Discovered all files in source directories"
            # Count unique directories from processed files
            unique_dirs = set(str(Path(p).parent) for p in self.state.processed_files.keys())
            summary_data = {
                'Total files found': self.state.files_scanned,
                'Unique directories': len(unique_dirs),
            }
            # Sample of files found
            sample_items = [
                {'path': k, 'type': v.get('file_type', 'unknown')}
                for k, v in list(self.state.processed_files.items())[:10]
            ]
            
        elif stage_name == "LLM_ANALYSIS":
            description = "Multi-stage LLM analysis: path analysis + regex generation + application"
            
            # Count files with extracted data
            files_with_class = sum(1 for v in self.state.processed_files.values() if v.get('taxonomic_class'))
            files_with_id = sum(1 for v in self.state.processed_files.values() if v.get('specimen_id'))
            files_with_year = sum(1 for v in self.state.processed_files.values() if v.get('campaign_year'))
            
            summary_data = {
                'Directories analyzed': len(self.state.directory_analyses),
                'LLM provider': config.llm_provider,
                'Files with taxonomic class': files_with_class,
                'Files with specimen ID': files_with_id,
                'Files with campaign year': files_with_year,
            }
            
            # Show sample of directory analyses
            sample_items = []
            for d, analysis in list(self.state.directory_analyses.items())[:10]:
                sample_items.append({
                    'directory': Path(d).name,
                    'class': analysis.get('taxonomic_class'),
                    'genus': analysis.get('determination'),
                    'regex_ok': analysis.get('regex_successes', 0),
                    'regex_fail': analysis.get('regex_failures', 0),
                })
            
        elif stage_name == "PATTERN_EXTRACTION":
            description = "Extracted specimen IDs, dates, and taxonomy from file paths"
            files_with_id = sum(1 for v in self.state.processed_files.values() if v.get('specimen_id'))
            summary_data = {
                'Files with specimen ID': files_with_id,
                'Files without ID': self.state.files_scanned - files_with_id,
            }
            sample_items = [
                {'file': v.get('filename'), 'specimen_id': v.get('specimen_id'), 'class': v.get('taxonomic_class')}
                for v in list(self.state.processed_files.values())[:10]
            ]
            
        elif stage_name == "HASHING":
            description = "Computed MD5 and perceptual hashes for duplicate detection"
            hashed = sum(1 for v in self.state.processed_files.values() if v.get('md5_hash'))
            summary_data = {
                'Files hashed': hashed,
            }
            sample_items = [
                {'file': v.get('filename'), 'md5': v.get('md5_hash', '')[:16] + '...' if v.get('md5_hash') else None}
                for v in list(self.state.processed_files.values())[:10]
                if v.get('md5_hash')
            ]
            
        elif stage_name == "DEDUPLICATION":
            description = "Identified duplicate files based on hash comparison"
            duplicates = sum(1 for v in self.state.processed_files.values() if v.get('is_duplicate'))
            summary_data = {
                'Duplicates found': duplicates,
                'Unique files': self.state.files_scanned - duplicates,
            }
            sample_items = [
                {'file': v.get('filename'), 'duplicate_of': v.get('duplicate_of')}
                for v in list(self.state.processed_files.values())[:10]
                if v.get('is_duplicate')
            ]
            
        elif stage_name == "ORGANIZING":
            description = "Moved/renamed files to organized folder structure"
            moved = sum(1 for v in self.state.processed_files.values() if v.get('moved'))
            summary_data = {
                'Files processed': moved if not self.dry_run else f"{moved} (dry run)",
                'Dry run': self.dry_run,
            }
            sample_items = [
                {'original': v.get('filename'), 'destination': v.get('destination_path')}
                for v in list(self.state.processed_files.values())[:10]
                if v.get('destination_path')
            ]
        
        # Request confirmation
        continue_pipeline = self.interaction_manager.confirm_stage(
            stage_name=stage_name,
            stage_description=description,
            summary_data=summary_data,
            sample_items=sample_items,
        )
        
        if not continue_pipeline:
            self._save_state()
            logger.info(f"Pipeline aborted after {stage_name}. State saved for resume.")
        
        return continue_pipeline

    def _run_scanning(self):
        """Stage 1: Scan directories to discover files."""
        logger.info("Stage 1: Scanning directories...")
        self.state.stage = PipelineStage.SCANNING
        
        all_files: List[ScannedFile] = []
        
        # Validate source directories
        valid_dirs = [d for d in self.source_dirs if d.exists()]
        if not valid_dirs:
            logger.warning("No valid source directories found")
            self.state.stage = PipelineStage.LLM_ANALYSIS
            self._save_state()
            return
        
        logger.info(f"Scanning {len(valid_dirs)} directories")
        
        # Use file scanner - scan_directories returns a generator
        for scanned_file in self.file_scanner.scan_directories(valid_dirs):
            all_files.append(scanned_file)
            self._report_progress("Scanning", len(all_files), self.file_scanner.progress.total_files)
        
        # Convert to ProcessedFile and store in state
        for sf in all_files:
            pf = ProcessedFile(
                original_path=sf.path,
                filename=sf.path.name,
                file_size=sf.file_size,
                file_type=sf.file_type,
            )
            self.state.processed_files[str(sf.path)] = pf.to_dict()
        
        self.state.total_files = len(all_files)
        self.state.files_scanned = len(all_files)
        self.state.stage = PipelineStage.LLM_ANALYSIS
        self._save_state()
        
        logger.info(f"Scanned {len(all_files)} files")
    
    def _run_llm_analysis(self):
        """
        Stage 2: Multi-stage LLM analysis of directories.
        
        Stage 2a: PATH ANALYSIS - Analyze directory paths only (no filenames)
                  Extracts: taxonomic_class, genus, campaign_year, collection_code
        
        Stage 2b: REGEX GENERATION - Generate regex patterns from sample filenames
                  Extracts: specimen_id regex patterns
        
        Stage 2c: REGEX APPLICATION - Apply regex to all files in each directory
                  Refines regex if partial failures
        
        Stage 2d: PER-FILE FALLBACK - For files where regex completely fails
                  Calls LLM per-file (user informed, can be slow)
        """
        logger.info("=" * 60)
        logger.info("Stage 2: MULTI-STAGE LLM ANALYSIS")
        logger.info("=" * 60)
        self.state.stage = PipelineStage.LLM_ANALYSIS
        
        # Get unique directories from processed files
        directories = set()
        dir_to_files = {}  # Map directory -> list of files
        for path_str in self.state.processed_files.keys():
            dir_path = str(Path(path_str).parent)
            directories.add(dir_path)
            if dir_path not in dir_to_files:
                dir_to_files[dir_path] = []
            dir_to_files[dir_path].append(path_str)
        
        directories = sorted(directories)
        total_dirs = len(directories)
        
        logger.info(f"Found {total_dirs} unique directories to analyze")
        print(f"\n{'='*60}")
        print(f"LLM ANALYSIS: {total_dirs} directories")
        print(f"{'='*60}")
        
        # Track statistics
        stats = {
            'path_analyses': 0,
            'regex_generated': 0,
            'regex_successes': 0,
            'regex_failures': 0,
            'per_file_fallbacks': 0,
        }
        
        for i, dir_path in enumerate(directories):
            # Skip if already analyzed
            if dir_path in self.state.directory_analyses:
                logger.info(f"[{i+1}/{total_dirs}] Skipping already analyzed: {dir_path}")
                continue
            
            print(f"\n--- Directory {i+1}/{total_dirs} ---")
            print(f"Path: {dir_path}")
            
            # Get all filenames from this directory
            all_filenames = [Path(p).name for p in dir_to_files.get(dir_path, [])]
            sample_filenames = all_filenames[:20]  # Max 20 for regex generation
            
            print(f"Files: {len(all_filenames)} (sample: {len(sample_filenames)})")
            
            # =========================================================
            # STAGE 2a: PATH ANALYSIS
            # =========================================================
            print(f"\n  [2a] PATH ANALYSIS...")
            try:
                # Use the client directly for path analysis
                client = self.directory_analyzer.client
                path_analysis = client.analyze_path(Path(dir_path))
                
                print(f"       Macroclass: {path_analysis.macroclass or 'None'}")
                print(f"       Taxonomic class: {path_analysis.taxonomic_class or 'None'}")
                print(f"       Genus/Order: {path_analysis.genus or 'None'}")
                print(f"       Collection: {path_analysis.collection_code or 'unknown'}")
                print(f"       Campaign year: {path_analysis.campaign_year or 'None'}")
                print(f"       Confidence: {path_analysis.confidence:.1%}")
                
                stats['path_analyses'] += 1
                
            except Exception as e:
                logger.warning(f"Path analysis failed for {dir_path}: {e}")
                print(f"       ERROR: {e}")
                path_analysis = None
            
            # =========================================================
            # STAGE 2b: REGEX GENERATION
            # =========================================================
            print(f"\n  [2b] REGEX GENERATION...")
            regex_result = None
            if sample_filenames:
                try:
                    regex_result = client.generate_filename_regex(
                        sample_filenames,
                        Path(dir_path)
                    )
                    
                    if regex_result.specimen_id_regex:
                        print(f"       Specimen ID regex: {regex_result.specimen_id_regex}")
                        print(f"       Extractable fields: {regex_result.extractable_fields}")
                        stats['regex_generated'] += 1
                    else:
                        print(f"       No regex pattern generated")
                        
                except Exception as e:
                    logger.warning(f"Regex generation failed for {dir_path}: {e}")
                    print(f"       ERROR: {e}")
            else:
                print(f"       Skipped (no files)")
            
            # =========================================================
            # STAGE 2c: REGEX APPLICATION
            # =========================================================
            successes = {}
            failures = []
            
            if regex_result and regex_result.specimen_id_regex:
                print(f"\n  [2c] APPLYING REGEX TO {len(all_filenames)} FILES...")
                try:
                    successes, failures, regex_result = client.apply_and_validate_regex(
                        regex_result,
                        all_filenames,
                        max_refinement_attempts=2
                    )
                    
                    success_rate = len(successes) / len(all_filenames) if all_filenames else 0
                    print(f"       Matched: {len(successes)}/{len(all_filenames)} ({success_rate:.1%})")
                    
                    if failures:
                        print(f"       Failed: {len(failures)} files")
                        for fn in failures[:3]:
                            print(f"         - {fn}")
                        if len(failures) > 3:
                            print(f"         ... and {len(failures) - 3} more")
                    
                    stats['regex_successes'] += len(successes)
                    stats['regex_failures'] += len(failures)
                    
                except Exception as e:
                    logger.warning(f"Regex application failed: {e}")
                    print(f"       ERROR: {e}")
                    failures = all_filenames
            else:
                failures = all_filenames
                print(f"\n  [2c] SKIPPED (no regex)")
            
            # =========================================================
            # STAGE 2d: PER-FILE FALLBACK (optional, inform user)
            # =========================================================
            file_analyses = {}
            if failures and len(failures) > 0:
                print(f"\n  [2d] PER-FILE FALLBACK...")
                
                # Only do per-file fallback for small numbers
                if len(failures) <= 20:
                    print(f"       Analyzing {len(failures)} files individually...")
                    
                    for j, fn in enumerate(failures):
                        try:
                            file_analysis = client.analyze_file(fn, str(dir_path))
                            file_analyses[fn] = file_analysis.to_dict()
                            stats['per_file_fallbacks'] += 1
                            
                            if file_analysis.specimen_id:
                                print(f"       [{j+1}/{len(failures)}] {fn} → {file_analysis.specimen_id}")
                            else:
                                print(f"       [{j+1}/{len(failures)}] {fn} → No ID extracted")
                                
                        except Exception as e:
                            logger.warning(f"Per-file analysis failed for {fn}: {e}")
                            
                else:
                    print(f"       SKIPPED ({len(failures)} files is too many for per-file analysis)")
                    print(f"       These files will need manual review or batch processing later")
            
            # =========================================================
            # STORE RESULTS
            # =========================================================
            # Build combined DirectoryAnalysis for backwards compatibility
            analysis_dict = {
                'directory_path': dir_path,
                'macroclass': path_analysis.macroclass if path_analysis else None,
                'taxonomic_class': path_analysis.taxonomic_class if path_analysis else None,
                'determination': path_analysis.genus if path_analysis else None,
                'collection_code': path_analysis.collection_code if path_analysis else 'unknown',
                'campaign_year': path_analysis.campaign_year if path_analysis else None,
                'specimen_id_regex': regex_result.specimen_id_regex if regex_result else None,
                'confidence': path_analysis.confidence if path_analysis else 0.0,
                'sample_filenames': sample_filenames[:10],
                # New fields for detailed tracking
                'regex_successes': len(successes),
                'regex_failures': len(failures),
                'per_file_analyses': len(file_analyses),
            }
            
            self.state.directory_analyses[dir_path] = analysis_dict
            
            # Update processed files with extracted data
            for path_str in dir_to_files.get(dir_path, []):
                filename = Path(path_str).name
                pf_data = self.state.processed_files.get(path_str, {})
                
                # Apply path-level metadata
                if path_analysis:
                    if path_analysis.macroclass:
                        pf_data['macroclass'] = path_analysis.macroclass
                    if path_analysis.taxonomic_class:
                        pf_data['taxonomic_class'] = path_analysis.taxonomic_class
                    if path_analysis.genus:
                        pf_data['determination'] = path_analysis.genus
                    if path_analysis.collection_code and path_analysis.collection_code != 'unknown':
                        pf_data['collection_code'] = path_analysis.collection_code
                    if path_analysis.campaign_year:
                        pf_data['campaign_year'] = path_analysis.campaign_year
                
                # Apply regex-extracted data
                if filename in successes:
                    extracted = successes[filename]
                    if 'specimen_id' in extracted:
                        pf_data['specimen_id'] = extracted['specimen_id']
                    if 'campaign_year' in extracted:
                        pf_data['campaign_year'] = extracted['campaign_year']
                
                # Apply per-file fallback data
                if filename in file_analyses:
                    fa = file_analyses[filename]
                    if fa.get('specimen_id') and not pf_data.get('specimen_id'):
                        pf_data['specimen_id'] = fa['specimen_id']
                    if fa.get('campaign_year') and not pf_data.get('campaign_year'):
                        pf_data['campaign_year'] = fa['campaign_year']
                
                self.state.processed_files[path_str] = pf_data
            
            self.state.directories_analyzed = i + 1
            self._report_progress("LLM Analysis", i + 1, total_dirs)
            
            # Save state after each directory
            self._save_state()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"LLM ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"  Path analyses: {stats['path_analyses']}")
        print(f"  Regex patterns generated: {stats['regex_generated']}")
        print(f"  Files matched by regex: {stats['regex_successes']}")
        print(f"  Files failed regex: {stats['regex_failures']}")
        print(f"  Per-file fallbacks: {stats['per_file_fallbacks']}")
        print()
        
        self.state.stage = PipelineStage.PATTERN_EXTRACTION
        self._save_state()
        
        logger.info(f"Analyzed {len(directories)} directories")
    
    def _run_pattern_extraction(self):
        """Stage 3: Extract patterns - known patterns first, then LLM regex if needed."""
        logger.info("Stage 3: Extracting patterns...")
        self.state.stage = PipelineStage.PATTERN_EXTRACTION
        
        total = len(self.state.processed_files)
        
        for i, (path_str, pf_data) in enumerate(self.state.processed_files.items()):
            path = Path(path_str)
            dir_path = str(path.parent)
            
            # Get directory analysis
            dir_analysis = self.state.directory_analyses.get(dir_path, {})
            
            # PRIORITY 1: Try known pattern extractor first (regex-based, no LLM)
            result = self.pattern_extractor.extract(path)
            if result.specimen_id:
                pf_data['specimen_id'] = result.specimen_id.specimen_id  # Use specimen_id attribute
                logger.debug(f"Specimen ID from known pattern: {result.specimen_id.specimen_id}")
            
            # PRIORITY 2: If no specimen ID from known patterns, try LLM-provided regex
            if not pf_data.get('specimen_id') and dir_analysis.get('specimen_id_regex'):
                analysis = DirectoryAnalysis.from_dict(dir_analysis)
                specimen_id = analysis.extract_specimen_id(path.name)
                if specimen_id:
                    pf_data['specimen_id'] = specimen_id
                    logger.debug(f"Specimen ID from LLM regex: {specimen_id}")
            
            # Apply directory-level metadata
            if dir_analysis.get('collection_code'):
                pf_data['collection_code'] = dir_analysis['collection_code']
            elif not pf_data.get('collection_code'):
                # Try to detect from path
                collection = detect_collection(path)
                if collection:
                    pf_data['collection_code'] = collection
            
            if dir_analysis.get('taxonomic_class'):
                pf_data['taxonomic_class'] = dir_analysis['taxonomic_class']
            
            if dir_analysis.get('determination'):
                pf_data['determination'] = dir_analysis['determination']
            
            if dir_analysis.get('campaign_year'):
                pf_data['campaign_year'] = dir_analysis['campaign_year']
            
            self._report_progress("Pattern Extraction", i + 1, total)
        
        self.state.stage = PipelineStage.HASHING
        self._save_state()
        
        logger.info("Pattern extraction complete")
    
    def _run_hashing(self):
        """Stage 4: Compute hashes for images."""
        logger.info("Stage 4: Computing hashes...")
        self.state.stage = PipelineStage.HASHING
        
        # Filter to only image files
        image_files = [
            (path_str, pf_data)
            for path_str, pf_data in self.state.processed_files.items()
            if pf_data.get('file_type') == FileType.IMAGE.value
        ]
        
        total = len(image_files)
        logger.info(f"Hashing {total} image files")
        
        for i, (path_str, pf_data) in enumerate(image_files):
            path = Path(path_str)
            
            if not path.exists():
                continue
            
            try:
                # Compute hashes and register
                hash_result = self.hash_registry.add(path)
                
                pf_data['md5_hash'] = hash_result.md5_hash
                pf_data['phash'] = hash_result.phash
                
                # Extract EXIF date if available (uses DateTimeOriginal)
                from .file_utils import MetadataExtractor
                metadata = MetadataExtractor.extract(path)
                if metadata and metadata.date_taken:
                    pf_data['fecha_captura'] = metadata.date_taken
                    # If no campaign_year from path, use EXIF year
                    if not pf_data.get('campaign_year') and metadata.datetime_original:
                        pf_data['campaign_year'] = metadata.datetime_original.year
                
            except Exception as e:
                logger.warning(f"Failed to hash {path}: {e}")
                pf_data['error'] = str(e)
            
            self.state.files_hashed = i + 1
            self._report_progress("Hashing", i + 1, total)
            
            # Save state periodically
            if (i + 1) % 100 == 0:
                self._save_state()
        
        self.state.stage = PipelineStage.DEDUPLICATION
        self._save_state()
        
        logger.info(f"Hashed {total} files")
    
    def _run_deduplication(self):
        """Stage 5: Identify duplicates."""
        logger.info("Stage 5: Identifying duplicates...")
        self.state.stage = PipelineStage.DEDUPLICATION
        
        # Find duplicate groups
        duplicate_groups_dict = self.hash_registry.get_all_duplicate_groups()
        
        # Flatten to list of groups
        all_groups = []
        for dup_type, groups in duplicate_groups_dict.items():
            all_groups.extend(groups)
        
        # Mark duplicates in processed files
        for group in all_groups:
            # First file is the "original"
            original_path = str(group.files[0])
            
            for dup_path in group.files[1:]:
                path_str = str(dup_path)
                if path_str in self.state.processed_files:
                    self.state.processed_files[path_str]['is_duplicate'] = True
                    self.state.processed_files[path_str]['duplicate_of'] = original_path
            
            # Store group info
            self.state.duplicate_groups.append({
                'type': group.duplicate_type.value,
                'hash': group.hash_value,
                'files': [str(f) for f in group.files],
            })
        
        self.state.stage = PipelineStage.REGISTRY_GENERATION  # Log to Excel FIRST
        self._save_state()
        
        dup_count = sum(1 for pf in self.state.processed_files.values() if pf.get('is_duplicate'))
        logger.info(f"Found {len(all_groups)} duplicate groups ({dup_count} duplicate files)")
    
    def _generate_new_filename(self, pf_data: Dict[str, Any], original_path: Path) -> str:
        """
        Generate new filename following the pattern: <campaign_year>_<specimen_id>_<uuid>.<ext>
        
        Args:
            pf_data: Processed file data dictionary
            original_path: Original file path
            
        Returns:
            New filename string
        """
        # Get or generate UUID
        file_uuid = pf_data.get('file_uuid')
        if not file_uuid:
            file_uuid = str(uuid_lib.uuid4())[:8]  # Use short UUID
            pf_data['file_uuid'] = file_uuid
        
        # Get campaign year (default to 0000 if unknown)
        campaign_year = pf_data.get('campaign_year')
        year_str = str(campaign_year) if campaign_year else "0000"
        
        # Get specimen ID (use normalized original name if not available)
        specimen_id = pf_data.get('specimen_id')
        if specimen_id:
            # Normalize: replace spaces with underscores, remove special chars
            specimen_id = re.sub(r'[^\w\-]', '_', specimen_id)
            specimen_id = re.sub(r'_+', '_', specimen_id).strip('_')
        else:
            # Use original filename stem, normalized
            stem = original_path.stem
            specimen_id = re.sub(r'[^\w\-]', '_', stem)
            specimen_id = re.sub(r'_+', '_', specimen_id).strip('_')
        
        # Build new filename
        extension = original_path.suffix.lower()
        new_filename = f"{year_str}_{specimen_id}_{file_uuid}{extension}"
        
        return new_filename
    
    def _get_image_destination(self, pf_data: Dict[str, Any], source_path: Path) -> Tuple[Path, str]:
        """
        Determine destination path for an image file.
        Structure: Campaña_[AÑO]/Macroclase/Clase/Genero/filename
        
        With fallbacks:
        - Sin_Campaña if no year
        - Sin_Macroclase if no macroclass
        - Sin_Clase if no class
        - Sin_Genero if no determination
        - Casos_Perdidos_Generales for files with major issues
        
        Args:
            pf_data: Processed file data
            source_path: Original file path
            
        Returns:
            Tuple of (destination_directory, new_filename)
        """
        # Handle duplicates separately
        if pf_data.get('is_duplicate'):
            dest_dir = self.output_dir / 'Duplicados'
            new_filename = self._generate_new_filename(pf_data, source_path)
            return dest_dir, new_filename
        
        # Handle items needing review
        if pf_data.get('needs_review'):
            reason = pf_data.get('review_reason', 'Otro')
            dest_dir = self.output_dir / 'Revision_Manual' / reason
            new_filename = self._generate_new_filename(pf_data, source_path)
            return dest_dir, new_filename
        
        # Check for critical missing data - goes to Casos_Perdidos_Generales
        has_any_taxonomy = pf_data.get('macroclass') or pf_data.get('taxonomic_class') or pf_data.get('determination')
        has_specimen = pf_data.get('specimen_id')
        
        if not has_any_taxonomy and not has_specimen:
            # Total loss - no taxonomy or specimen info
            dest_dir = self.output_dir / 'Casos_Perdidos_Generales'
            new_filename = self._generate_new_filename(pf_data, source_path)
            return dest_dir, new_filename
        
        # Start building destination path
        dest_dir = self.output_dir
        
        # Campaign level
        campaign = pf_data.get('campaign_year')
        if campaign:
            dest_dir = dest_dir / f"Campaña_{campaign}"
        else:
            dest_dir = dest_dir / 'Sin_Campaña'
        
        # Macroclass level
        macroclass = pf_data.get('macroclass')
        if not macroclass:
            # Try to infer from taxonomic_class
            tax_class = pf_data.get('taxonomic_class')
            macroclass = get_macroclass_folder(tax_class)
        
        if macroclass and macroclass != 'Unsorted_Macroclass':
            dest_dir = dest_dir / macroclass
        else:
            dest_dir = dest_dir / 'Sin_Macroclase'
        
        # Class level
        tax_class = pf_data.get('taxonomic_class')
        if tax_class:
            dest_dir = dest_dir / tax_class
        else:
            dest_dir = dest_dir / 'Sin_Clase'
        
        # Determination/Genus level (optional)
        determination = pf_data.get('determination')
        if determination:
            dest_dir = dest_dir / determination
        else:
            dest_dir = dest_dir / 'Sin_Genero'
        
        # Generate new filename
        new_filename = self._generate_new_filename(pf_data, source_path)
        
        return dest_dir, new_filename
    
    def _get_nonimage_destination(self, pf_data: Dict[str, Any], source_path: Path, is_text: bool) -> Tuple[Path, str]:
        """
        Determine destination path for non-image files (text or other).
        Pattern: <uuid>_<original_filename>.<ext>
        
        Args:
            pf_data: Processed file data
            source_path: Original file path
            is_text: True for text files, False for other files
            
        Returns:
            Tuple of (destination_directory, new_filename)
        """
        # Get or generate UUID
        file_uuid = pf_data.get('file_uuid')
        if not file_uuid:
            file_uuid = str(uuid_lib.uuid4())[:8]
            pf_data['file_uuid'] = file_uuid
        
        # Determine base directory
        if is_text:
            base_dir = self.output_dir / 'Annotations'
        else:
            base_dir = self.output_dir / 'Other_Files'
        
        # Optional: split by collection if known
        collection = pf_data.get('collection_code')
        if collection:
            dest_dir = base_dir / collection
        else:
            dest_dir = base_dir / 'Unsorted'
        
        # Generate new filename: <uuid>_<original_name>
        original_name = source_path.name
        # Sanitize original name
        safe_name = re.sub(r'[^\w\.\-]', '_', original_name)
        new_filename = f"{file_uuid}_{safe_name}"
        
        return dest_dir, new_filename
    
    def _run_organizing(self):
        """Stage 7: Organize files into collection folders with renaming.
        
        NOTE: This runs AFTER registry generation, so Excel logs are complete
        even if file move fails.
        """
        logger.info("Stage 7: Organizing files (moving to final destinations)...")
        self.state.stage = PipelineStage.ORGANIZING
        
        if self.dry_run:
            logger.info("DRY RUN - Not moving files")
        
        # Create base output directories
        base_dirs = [
            self.output_dir / 'Las_Hoyas',
            self.output_dir / 'Buenache',
            self.output_dir / 'Montsec',
            self.output_dir / 'Unknown_Collection',
            self.output_dir / 'Duplicates',
            self.output_dir / 'Manual_Review',
            self.output_dir / 'Annotations',
            self.output_dir / 'Other_Files',
        ]
        
        if not self.dry_run:
            for dir_path in base_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        total = len(self.state.processed_files)
        
        for i, (path_str, pf_data) in enumerate(self.state.processed_files.items()):
            source_path = Path(path_str)
            
            if not source_path.exists():
                continue
            
            file_type_str = pf_data.get('file_type')
            file_type = FileType(file_type_str) if file_type_str else FileType.OTHER
            
            # Determine destination based on file type
            if file_type == FileType.IMAGE:
                dest_dir, new_filename = self._get_image_destination(pf_data, source_path)
            elif file_type == FileType.TEXT:
                dest_dir, new_filename = self._get_nonimage_destination(pf_data, source_path, is_text=True)
            else:
                dest_dir, new_filename = self._get_nonimage_destination(pf_data, source_path, is_text=False)
            
            dest_path = dest_dir / new_filename
            
            # Handle name conflicts
            if dest_path.exists():
                stem = Path(new_filename).stem
                suffix = Path(new_filename).suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                new_filename = dest_path.name
            
            pf_data['destination_path'] = str(dest_path)
            pf_data['new_filename'] = new_filename
            
            # Move/copy file
            if not self.dry_run:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    pf_data['moved'] = True
                    
                    # Log the action
                    self.action_logger.file_copied(source_path, dest_path)
                except Exception as e:
                    logger.warning(f"Failed to copy {source_path}: {e}")
                    pf_data['error'] = str(e)
            
            self.state.files_organized = i + 1
            self._report_progress("Organizing", i + 1, total)
            
            # Save state periodically
            if (i + 1) % 100 == 0:
                self._save_state()
        
        # Save deferred decisions log
        if self.interaction_manager.deferred_items:
            deferred_log_path = self.output_dir / "deferred_decisions.json"
            self.interaction_manager.save_deferred_log(deferred_log_path)
        
        # Clean up empty directories in source (if not dry run)
        if not self.dry_run:
            self._cleanup_empty_directories()
        
        self.state.stage = PipelineStage.COMPLETED  # Organizing is now the last stage
        self._save_state()
        
        logger.info(f"Organized {self.state.files_organized} files")
    
    def _cleanup_empty_directories(self):
        """Remove empty directories from source after moving files."""
        from .file_utils import FileOperations
        ops = FileOperations()
        
        deleted_count = 0
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                continue
            
            # Walk in reverse order (deepest first) to handle nested empty dirs
            for dirpath in sorted(source_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dirpath.is_dir():
                    try:
                        if ops.delete_empty_directory(dirpath):
                            deleted_count += 1
                            self.action_logger.dir_empty_deleted(dirpath)
                    except Exception as e:
                        logger.debug(f"Could not delete {dirpath}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} empty directories")
    
    def _run_registry_generation(self):
        """Stage 6: Generate Excel registries for images, text files, and other files.
        
        IMPORTANT: This runs BEFORE organizing (file move) so that if move fails,
        we still have a complete log of intended operations.
        """
        logger.info("Stage 6: Generating registries (logging to Excel BEFORE file move)...")
        self.state.stage = PipelineStage.REGISTRY_GENERATION
        
        # Create Excel manager
        registries_dir = self.output_dir / "registries"
        registries_dir.mkdir(parents=True, exist_ok=True)
        excel_manager = ExcelManager(registries_dir)
        
        # Counters for different file types
        image_count = 0
        text_count = 0
        other_count = 0
        
        # Add processed files to appropriate registries
        for path_str, pf_data in self.state.processed_files.items():
            file_type_str = pf_data.get('file_type')
            file_type = FileType(file_type_str) if file_type_str else FileType.OTHER
            
            if file_type == FileType.IMAGE:
                # Build comment with auto-extraction info
                comments = []
                if pf_data.get('is_duplicate'):
                    comments.append(f"[AUTO] duplicate_of={pf_data.get('duplicate_of')}")
                if pf_data.get('needs_review'):
                    comments.append(f"[REVIEW] {pf_data.get('review_reason')}")
                if pf_data.get('collection_code'):
                    comments.append(f"[AUTO] collection={pf_data.get('collection_code')}")
                if pf_data.get('specimen_id'):
                    comments.append(f"[AUTO] specimen_id extracted from path/filename")
                
                record = ImageRecord(
                    uuid=pf_data.get('file_uuid') or ImageRecord.generate_uuid(),
                    specimen_id=pf_data.get('specimen_id'),
                    original_path=path_str,
                    current_path=pf_data.get('destination_path'),
                    macroclass_label=pf_data.get('macroclass') or get_macroclass_folder(pf_data.get('taxonomic_class')),
                    class_label=pf_data.get('taxonomic_class'),
                    genera_label=pf_data.get('determination'),
                    fecha_captura=pf_data.get('fecha_captura'),  # EXIF DateTimeOriginal
                    campaign_year=str(pf_data.get('campaign_year')) if pf_data.get('campaign_year') else None,
                    fuente=pf_data.get('collection_code'),
                    hash_perceptual=pf_data.get('phash'),
                    comentarios=" | ".join(comments) if comments else "[AUTO] processed by pipeline",
                )
                excel_manager.add_image(record)
                image_count += 1
                
            elif file_type == FileType.TEXT:
                record = TextFileRecord(
                    id=pf_data.get('file_uuid') or str(uuid_lib.uuid4())[:8],
                    original_path=path_str,
                    current_path=pf_data.get('destination_path') or '',
                    original_filename=pf_data.get('filename', ''),
                    file_type=Path(path_str).suffix,
                    processed=False,
                    extracted_info=None,
                )
                excel_manager.add_text_file(record)
                text_count += 1
                
            else:
                record = OtherFileRecord(
                    id=pf_data.get('file_uuid') or str(uuid_lib.uuid4())[:8],
                    original_path=path_str,
                    current_path=pf_data.get('destination_path') or '',
                    original_filename=pf_data.get('filename', ''),
                    file_type=Path(path_str).suffix,
                )
                excel_manager.add_other_file(record)
                other_count += 1
        
        # Save all registries
        excel_manager.save_all()
        logger.info(f"Saved registries to: {registries_dir}")
        logger.info(f"  Images: {image_count}, Text files: {text_count}, Other: {other_count}")
        
        # Also save a summary CSV for easy viewing
        summary_path = self.output_dir / "processing_summary.csv"
        self._save_summary_csv(summary_path)
        logger.info(f"Saved summary: {summary_path}")
        
        # Transition to ORGANIZING stage (file move)
        self.state.stage = PipelineStage.ORGANIZING
        self._save_state()
        
        logger.info("Registry generation complete - proceeding to file move")
    
    def _save_summary_csv(self, path: Path):
        """Save a summary CSV of all processed files."""
        import csv
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'original_path', 'filename', 'specimen_id', 'collection',
                'taxonomic_class', 'determination', 'campaign_year',
                'is_duplicate', 'destination_path', 'md5_hash'
            ])
            
            for path_str, pf_data in self.state.processed_files.items():
                writer.writerow([
                    path_str,
                    pf_data.get('filename', ''),
                    pf_data.get('specimen_id', ''),
                    pf_data.get('collection_code', ''),
                    pf_data.get('taxonomic_class', ''),
                    pf_data.get('determination', ''),
                    pf_data.get('campaign_year', ''),
                    pf_data.get('is_duplicate', False),
                    pf_data.get('destination_path', ''),
                    pf_data.get('md5_hash', ''),
                ])
    
    def get_summary(self) -> dict:
        """Get a summary of the pipeline results."""
        # Count by collection
        by_collection = {}
        by_class = {}
        duplicates = 0
        errors = 0
        
        for pf_data in self.state.processed_files.values():
            collection = pf_data.get('collection_code', 'unknown')
            by_collection[collection] = by_collection.get(collection, 0) + 1
            
            tax_class = pf_data.get('taxonomic_class', 'Unknown')
            by_class[tax_class] = by_class.get(tax_class, 0) + 1
            
            if pf_data.get('is_duplicate'):
                duplicates += 1
            
            if pf_data.get('error'):
                errors += 1
        
        return {
            'stage': self.state.stage.value,
            'total_files': self.state.total_files,
            'files_organized': self.state.files_organized,
            'directories_analyzed': self.state.directories_analyzed,
            'duplicates': duplicates,
            'errors': errors,
            'by_collection': by_collection,
            'by_taxonomic_class': by_class,
            'started_at': self.state.started_at,
            'completed_at': self.state.completed_at,
        }


def run_pipeline(
    source_dirs: List[Path | str],
    output_dir: Path | str = None,
    use_llm: bool = True,
    dry_run: bool = False,
    resume: bool = False,
    progress_callback: Callable = None
) -> PipelineState:
    """
    Convenience function to run the full pipeline.
    
    Args:
        source_dirs: Directories to process
        output_dir: Output directory
        use_llm: Whether to use LLM for analysis
        dry_run: If True, don't move files
        resume: If True, try to resume from saved state
        progress_callback: Progress callback function
        
    Returns:
        Final PipelineState
    """
    source_paths = [Path(d) for d in source_dirs]
    output_path = Path(output_dir) if output_dir else None
    
    orchestrator = PipelineOrchestrator(
        source_dirs=source_paths,
        output_dir=output_path,
        use_llm=use_llm,
        dry_run=dry_run,
        progress_callback=progress_callback
    )
    
    return orchestrator.run(resume=resume)
