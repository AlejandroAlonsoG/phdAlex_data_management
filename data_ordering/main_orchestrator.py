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
    create_duplicate_decision, create_duplicate_metadata_decision,
    create_unknown_collection_decision,
    create_source_discrepancy_decision, create_camera_number_decision,
    SOURCE_LABELS,
)


logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Stages of the processing pipeline."""
    INIT = "init"
    SCANNING = "scanning"
    LLM_ANALYSIS = "llm_analysis"
    PATTERN_EXTRACTION = "pattern_extraction"
    METADATA_RECONCILIATION = "metadata_reconciliation"
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
       2a. PATH ANALYSIS — extract taxonomy, year, collection from directory path
       2b. REGEX GENERATION — LLM generates regex for filenames
       2c. REGEX APPLICATION — apply regex to every file
       2d. PER-FILE FALLBACK — individual LLM call for unmatched files
    3. PATTERN_EXTRACTION: Extract specimen IDs using predefined heuristic patterns
    4. METADATA_RECONCILIATION: Compare metadata from all sources (Path LLM,
       Filename LLM, Pattern Extractor), detect discrepancies, prompt the user
       to resolve them (with option to apply same resolution to whole
       subdirectory).  Also handles camera-number flags.
    5. HASHING: Compute MD5 and perceptual hashes for images
    6. DEDUPLICATION: Identify and group duplicates
    7. ORGANIZING: Move files to collection-based folders (with renaming)
    8. REGISTRY_GENERATION: Create Excel registries
    
    File handling:
    - Images: Las Hoyas (default) at root: Campaign/Macroclass/Class/Determination
             Other collections under Otras_Colecciones/<Collection>/Campaign/...
    - Text files: Moved to Archivos_Texto folder, renamed with UUID prefix
    - Other files: Moved to Otros_Archivos folder, renamed with UUID prefix
    """
    
    @staticmethod
    def compute_staging_dir(output_dir: Path, source_dirs: List[Path]) -> Path:
        """
        Compute the staging directory name: <output_dir>_<source_dir_name>.
        
        If multiple source dirs, joins their names with '_'.
        
        Args:
            output_dir: The final output directory
            source_dirs: List of source directories being processed
            
        Returns:
            Path to the staging directory
        """
        if not source_dirs:
            return output_dir / "_staging"
        
        # Build a suffix from source directory names
        source_names = []
        for sd in source_dirs:
            name = Path(sd).name
            # Sanitize: remove special chars, keep alphanumeric and underscores
            safe_name = re.sub(r'[^\w\-]', '_', name)
            safe_name = re.sub(r'_+', '_', safe_name).strip('_')
            if safe_name:
                source_names.append(safe_name)
        
        suffix = '_'.join(source_names) if source_names else 'unknown'
        staging_name = f"{output_dir.name}_{suffix}"
        return output_dir.parent / staging_name

    def __init__(
        self,
        source_dirs: List[Path],
        output_dir: Path = None,
        output_base: Path = None,  # Alias for output_dir
        state_path: Path = None,
        use_llm: bool = True,
        interactive: bool = False,  # Legacy parameter
        interaction_mode: str = None,  # New: 'interactive', 'deferred', 'auto_accept', 'step_by_step'
        progress_callback: Callable[[str, int, int], None] = None,
        phash_threshold: int = 8,
        use_staging: bool = True,  # New: whether to use staging directory
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
            use_staging: If True, output to staging dir <output>_<source> for validation
        """
        dry_run = False

        self.source_dirs = [Path(d) for d in source_dirs]
        self.final_output_dir = Path(output_base or output_dir) if (output_base or output_dir) else config.output_base_dir
        self.use_staging = use_staging
        
        if use_staging and self.source_dirs:
            self.output_dir = self.compute_staging_dir(self.final_output_dir, self.source_dirs)
        else:
            self.output_dir = self.final_output_dir
        
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
        # FileScanner only discovers & classifies files; hashing, pattern
        # extraction, and deduplication are handled by dedicated pipeline stages.
        self.file_scanner = FileScanner(hash_images=False, extract_patterns=False)
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
                    (provider == 'local' and OPENAI_AVAILABLE) or
                    (provider == 'gemini' and GENAI_AVAILABLE) or
                    (provider not in ('github', 'gemini', 'local') and (GENAI_AVAILABLE or OPENAI_AVAILABLE))
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
        
        # Save staging info so the merge tool knows the final destination
        if self.use_staging and self.output_dir != self.final_output_dir:
            staging_info = {
                'staging_dir': str(self.output_dir),
                'final_output_dir': str(self.final_output_dir),
                'source_directories': [str(d) for d in self.source_dirs],
                'created_at': datetime.now().isoformat(),
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            staging_info_path = self.output_dir / "staging_info.json"
            with open(staging_info_path, 'w', encoding='utf-8') as f:
                json.dump(staging_info, f, indent=2, ensure_ascii=False)
            logger.info(f"Staging directory: {self.output_dir}")
            logger.info(f"Final output will be merged to: {self.final_output_dir}")
        
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
            
            if self.state.stage == PipelineStage.METADATA_RECONCILIATION:
                self._run_metadata_reconciliation()
                if not self._confirm_stage_completion("METADATA_RECONCILIATION"):
                    return self.state
            
            if self.state.stage == PipelineStage.HASHING:
                self._run_hashing()
                if not self._confirm_stage_completion("HASHING"):
                    return self.state
            
            if self.state.stage == PipelineStage.DEDUPLICATION:
                self._run_deduplication()
                if not self._confirm_stage_completion("DEDUPLICATION"):
                    return self.state
            
            # IMPORTANT: Move files FIRST, then generate registries
            # so that current_path is populated in the registry
            if self.state.stage == PipelineStage.ORGANIZING:
                self._run_organizing()
                if not self._confirm_stage_completion("ORGANIZING"):
                    return self.state
            
            if self.state.stage == PipelineStage.REGISTRY_GENERATION:
                self._run_registry_generation()
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
            
        elif stage_name == "METADATA_RECONCILIATION":
            description = "Compared metadata from Path LLM / Filename LLM / Pattern Extractor and resolved discrepancies"
            needs_review = sum(1 for v in self.state.processed_files.values() if v.get('needs_review'))
            # Count files that had metadata_sources with >1 source on any field
            files_with_multi = 0
            for v in self.state.processed_files.values():
                msrc = v.get('metadata_sources', {})
                if any(len(src_vals) > 1 for src_vals in msrc.values()):
                    files_with_multi += 1
            summary_data = {
                'Files with multi-source metadata': files_with_multi,
                'Files flagged for review': needs_review,
            }
            sample_items = [
                {
                    'file': v.get('filename'),
                    'specimen_id': v.get('specimen_id'),
                    'campaign_year': v.get('campaign_year'),
                    'class': v.get('taxonomic_class'),
                    'needs_review': v.get('needs_review', False),
                }
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
        self.action_logger.stage_start("SCANNING", f"Source dirs: {[str(d) for d in self.source_dirs]}")
        self.state.stage = PipelineStage.SCANNING
        
        all_files: List[ScannedFile] = []
        
        # Validate source directories
        valid_dirs = [d for d in self.source_dirs if d.exists()]
        if not valid_dirs:
            logger.warning("No valid source directories found")
            self.action_logger.warning("No valid source directories found")
            self.state.stage = PipelineStage.LLM_ANALYSIS
            self._save_state()
            return
        
        logger.info(f"Scanning {len(valid_dirs)} directories")
        self.action_logger.info(f"Scanning {len(valid_dirs)} directories: {[str(d) for d in valid_dirs]}")
        
        # Use file scanner - scan_directories returns a generator
        for scanned_file in self.file_scanner.scan_directories(valid_dirs):
            all_files.append(scanned_file)
            # Log each discovered file
            self.action_logger.file_scanned(
                scanned_file.path,
                scanned_file.file_type.value if scanned_file.file_type else 'unknown',
                scanned_file.file_size
            )
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
        
        # Log type breakdown
        type_counts = {}
        for sf in all_files:
            t = sf.file_type.value if sf.file_type else 'unknown'
            type_counts[t] = type_counts.get(t, 0) + 1
        self.action_logger.stage_end("SCANNING", f"Total: {len(all_files)} files | Breakdown: {type_counts}")
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
        self.action_logger.stage_start("LLM_ANALYSIS", f"Provider: {config.llm_provider}")
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
                
                # Log to action_logger
                self.action_logger.llm_path_analysis(
                    directory=dir_path,
                    macroclass=path_analysis.macroclass or 'None',
                    taxonomic_class=path_analysis.taxonomic_class or 'None',
                    genus=path_analysis.genus or 'None',
                    collection=path_analysis.collection_code or 'unknown',
                    campaign_year=path_analysis.campaign_year,
                    confidence=path_analysis.confidence
                )
                
                stats['path_analyses'] += 1
                
            except Exception as e:
                logger.warning(f"Path analysis failed for {dir_path}: {e}")
                self.action_logger.llm_error("path_analysis", dir_path, str(e))
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
                        self.action_logger.llm_regex_generated(
                            directory=dir_path,
                            regex=regex_result.specimen_id_regex,
                            extractable_fields=regex_result.extractable_fields
                        )
                        stats['regex_generated'] += 1
                    else:
                        print(f"       No regex pattern generated")
                        self.action_logger.info(f"No regex pattern generated for {dir_path}")
                        
                except Exception as e:
                    logger.warning(f"Regex generation failed for {dir_path}: {e}")
                    self.action_logger.llm_error("regex_generation", dir_path, str(e))
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
                    
                    # Log regex application results
                    self.action_logger.llm_regex_applied(
                        directory=dir_path,
                        total=len(all_filenames),
                        matched=len(successes),
                        failed=len(failures)
                    )
                    
                    if failures:
                        print(f"       Failed: {len(failures)} files")
                        for fn in failures[:3]:
                            print(f"         - {fn}")
                            self.action_logger.specimen_id_missing(fn, reason="regex_mismatch")
                        if len(failures) > 3:
                            print(f"         ... and {len(failures) - 3} more")
                    
                    # Log successful extractions
                    for fn, extracted in successes.items():
                        if 'specimen_id' in extracted:
                            self.action_logger.specimen_id_found(
                                fn, extracted['specimen_id'], source="llm_regex"
                            )
                    
                    stats['regex_successes'] += len(successes)
                    stats['regex_failures'] += len(failures)
                    
                except Exception as e:
                    logger.warning(f"Regex application failed: {e}")
                    self.action_logger.llm_error("regex_application", dir_path, str(e))
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
                                self.action_logger.llm_per_file(
                                    fn, file_analysis.specimen_id,
                                    f"class={file_analysis.taxonomic_class}, confidence={file_analysis.confidence:.1%}"
                                )
                            else:
                                print(f"       [{j+1}/{len(failures)}] {fn} → No ID extracted")
                                self.action_logger.llm_per_file(fn, None, "No ID extracted")
                                
                        except Exception as e:
                            logger.warning(f"Per-file analysis failed for {fn}: {e}")
                            self.action_logger.llm_error("per_file_analysis", fn, str(e))
                            
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
                
                # Ensure metadata_sources dict exists (field -> {source -> value})
                if 'metadata_sources' not in pf_data:
                    pf_data['metadata_sources'] = {}
                msrc = pf_data['metadata_sources']
                
                # --- Helper to record a value from a given source ---
                def _record(field: str, value, source: str):
                    """Store *value* under *field* for *source* and set the
                    top-level pf_data[field] (last-writer-wins for now; the
                    reconciliation stage will resolve conflicts)."""
                    if value is None:
                        return
                    if field not in msrc:
                        msrc[field] = {}
                    msrc[field][source] = value
                    pf_data[field] = value
                
                # Apply path-level metadata (source: path_llm)
                if path_analysis:
                    _record('macroclass', path_analysis.macroclass, 'path_llm')
                    _record('taxonomic_class', path_analysis.taxonomic_class, 'path_llm')
                    _record('determination', path_analysis.genus, 'path_llm')
                    if path_analysis.collection_code and path_analysis.collection_code != 'unknown':
                        _record('collection_code', path_analysis.collection_code, 'path_llm')
                    _record('campaign_year', path_analysis.campaign_year, 'path_llm')
                
                # Apply regex-extracted data (source: filename_llm)
                if filename in successes:
                    extracted = successes[filename]
                    if 'specimen_id' in extracted:
                        _record('specimen_id', extracted['specimen_id'], 'filename_llm')
                    if 'campaign_year' in extracted:
                        _record('campaign_year', extracted['campaign_year'], 'filename_llm')
                    # Carry over any other fields the regex extracted
                    for ex_field in ('macroclass', 'taxonomic_class', 'determination'):
                        if ex_field in extracted:
                            _record(ex_field, extracted[ex_field], 'filename_llm')
                
                # Apply per-file fallback data (source: filename_llm — same logical source)
                if filename in file_analyses:
                    fa = file_analyses[filename]
                    if fa.get('specimen_id'):
                        _record('specimen_id', fa['specimen_id'], 'filename_llm')
                    if fa.get('campaign_year'):
                        _record('campaign_year', fa['campaign_year'], 'filename_llm')
                    if fa.get('taxonomic_class'):
                        _record('taxonomic_class', fa['taxonomic_class'], 'filename_llm')
                    if fa.get('determination') or fa.get('genus'):
                        _record('determination', fa.get('determination') or fa.get('genus'), 'filename_llm')
                
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
        
        self.action_logger.stage_end(
            "LLM_ANALYSIS",
            f"Dirs: {stats['path_analyses']} | Regex: {stats['regex_generated']} | "
            f"Matched: {stats['regex_successes']} | Failed: {stats['regex_failures']} | "
            f"Fallbacks: {stats['per_file_fallbacks']}"
        )
        logger.info(f"Analyzed {len(directories)} directories")
    
    def _run_pattern_extraction(self):
        """Stage 3: Extract patterns - known patterns first, then LLM regex if needed.
        
        Records all extracted values into pf_data['metadata_sources'] so that
        the METADATA_RECONCILIATION stage can compare them and ask the user to
        resolve discrepancies.
        """
        logger.info("Stage 3: Extracting patterns...")
        self.action_logger.stage_start("PATTERN_EXTRACTION", f"{len(self.state.processed_files)} files to process")
        self.state.stage = PipelineStage.PATTERN_EXTRACTION
        
        total = len(self.state.processed_files)
        ids_from_known = 0
        ids_from_llm = 0
        ids_missing = 0
        camera_flags = 0
        
        for i, (path_str, pf_data) in enumerate(self.state.processed_files.items()):
            path = Path(path_str)
            dir_path = str(path.parent)
            
            # Ensure metadata_sources dict exists
            if 'metadata_sources' not in pf_data:
                pf_data['metadata_sources'] = {}
            msrc = pf_data['metadata_sources']
            
            def _record_pe(field: str, value):
                """Record a value from the pattern_extractor source."""
                if value is None:
                    return
                if field not in msrc:
                    msrc[field] = {}
                msrc[field]['pattern_extractor'] = value
                pf_data[field] = value
            
            # Get directory analysis
            dir_analysis = self.state.directory_analyses.get(dir_path, {})
            
            # Run the full heuristic extractor
            result = self.pattern_extractor.extract(path)
            
            # --- Specimen ID ---
            if result.specimen_id:
                _record_pe('specimen_id', result.specimen_id.specimen_id)
                logger.debug(f"Specimen ID from known pattern: {result.specimen_id.specimen_id}")
                self.action_logger.specimen_id_found(
                    path.name, result.specimen_id.specimen_id, source="known_pattern"
                )
                ids_from_known += 1
            
            # PRIORITY 2: If no specimen ID from known patterns, try LLM-provided regex
            if not pf_data.get('specimen_id') and dir_analysis.get('specimen_id_regex'):
                analysis = DirectoryAnalysis.from_dict(dir_analysis)
                specimen_id = analysis.extract_specimen_id(path.name)
                if specimen_id:
                    pf_data['specimen_id'] = specimen_id
                    logger.debug(f"Specimen ID from LLM regex: {specimen_id}")
                    self.action_logger.specimen_id_found(
                        path.name, specimen_id, source="llm_regex"
                    )
                    ids_from_llm += 1
            
            # Log if still no specimen ID
            if not pf_data.get('specimen_id'):
                ids_missing += 1
                self.action_logger.specimen_id_missing(
                    path.name,
                    reason="no_known_pattern_and_no_llm_regex"
                )
            
            # --- Camera-number flags ---
            # Store numeric IDs flagged as camera-generated so reconciliation
            # can prompt the user.
            cam_ids = [n for n in result.numeric_ids if n.is_likely_camera_number]
            if cam_ids:
                camera_flags += 1
                pf_data['camera_number_flags'] = [
                    {'numeric_id': n.numeric_id, 'raw_match': n.raw_match}
                    for n in cam_ids
                ]
            
            # --- Campaign year from heuristic date extraction ---
            best_date = result.get_best_date()
            if best_date:
                _record_pe('campaign_year', best_date.year)
            
            # --- Taxonomy hints from path ---
            for hint in result.taxonomy_hints:
                if hint.level == 'class':
                    _record_pe('taxonomic_class', hint.value)
                elif hint.level in ('genus', 'order', 'family'):
                    _record_pe('determination', hint.value)
            
            # --- Campaign from path analyzer ---
            if result.campaign and not pf_data.get('campaign_year'):
                _record_pe('campaign_year', result.campaign.year)
            
            # Apply directory-level metadata (these are already recorded as
            # path_llm in the LLM stage; no need to re-record source here,
            # just ensure the top-level field is populated).
            if dir_analysis.get('collection_code'):
                pf_data['collection_code'] = dir_analysis['collection_code']
            elif not pf_data.get('collection_code'):
                collection = detect_collection(path)
                if collection:
                    pf_data['collection_code'] = collection
            
            if pf_data.get('collection_code'):
                self.action_logger.collection_detected(
                    path.name, pf_data['collection_code'],
                    source="dir_analysis" if dir_analysis.get('collection_code') else "path_detection"
                )
            
            if dir_analysis.get('taxonomic_class') and not pf_data.get('taxonomic_class'):
                pf_data['taxonomic_class'] = dir_analysis['taxonomic_class']
            
            if dir_analysis.get('determination') and not pf_data.get('determination'):
                pf_data['determination'] = dir_analysis['determination']
            
            if dir_analysis.get('campaign_year') and not pf_data.get('campaign_year'):
                pf_data['campaign_year'] = dir_analysis['campaign_year']
            
            # Log full metadata summary for this file
            self.action_logger.metadata_extracted(
                path.name,
                specimen_id=pf_data.get('specimen_id'),
                collection=pf_data.get('collection_code'),
                taxonomy=pf_data.get('taxonomic_class'),
                campaign_year=pf_data.get('campaign_year'),
                source="pattern_extraction"
            )
            
            self._report_progress("Pattern Extraction", i + 1, total)
        
        self.state.stage = PipelineStage.METADATA_RECONCILIATION
        self._save_state()
        
        self.action_logger.stage_end(
            "PATTERN_EXTRACTION",
            f"IDs from known patterns: {ids_from_known} | IDs from LLM regex: {ids_from_llm} | "
            f"Missing: {ids_missing} | Camera flags: {camera_flags}"
        )
        logger.info("Pattern extraction complete")
    
    # ------------------------------------------------------------------
    # Fields eligible for source-discrepancy reconciliation
    # ------------------------------------------------------------------
    RECONCILABLE_FIELDS = (
        'specimen_id', 'campaign_year', 'macroclass',
        'taxonomic_class', 'determination',
    )

    # Priority ordering for automatic (non-interactive) conflict resolution.
    # First source in the list wins.  Sources not listed are ignored.
    FIELD_SOURCE_PRIORITY: Dict[str, List[str]] = {
        'specimen_id':     ['pattern_extractor', 'filename_llm', 'path_llm'],
        'campaign_year':   ['path_llm', 'pattern_extractor', 'filename_llm'],
        'macroclass':      ['path_llm', 'filename_llm', 'pattern_extractor'],
        'taxonomic_class': ['path_llm', 'filename_llm', 'pattern_extractor'],
        'determination':   ['path_llm', 'filename_llm', 'pattern_extractor'],
    }

    def _auto_resolve_discrepancy(
        self,
        field_name: str,
        source_values: Dict[str, Any],
    ) -> str:
        """Pick the winning source key for *field_name* using FIELD_SOURCE_PRIORITY.

        Returns the source key whose value should be used.  If none of the
        sources appear in the priority list the first source is returned as
        a fallback.
        """
        priority = self.FIELD_SOURCE_PRIORITY.get(field_name, [])
        for src in priority:
            if src in source_values and source_values[src] is not None:
                return src
        # Fallback: first available source
        return next(iter(source_values))

    def _run_metadata_reconciliation(self):
        """Stage 4: Reconcile metadata from different extraction sources.
        
        For every file, compares the values contributed by each source
        (path_llm, filename_llm, pattern_extractor).  When two or more
        sources disagree on a field:
        
        * **Interactive / step-by-step modes** → the user is prompted to
          pick the authoritative value (with an option to apply the same
          choice to the whole subdirectory).
        * **Non-interactive modes** (deferred, auto_accept) → the conflict
          is resolved automatically using ``FIELD_SOURCE_PRIORITY`` which
          defines a per-field ordering of trusted sources.
        
        Camera-number flags follow the same logic: interactive prompts the
        user, non-interactive discards them (safer default).
        """
        logger.info("Stage 4: Reconciling metadata sources...")
        self.action_logger.stage_start(
            "METADATA_RECONCILIATION",
            f"{len(self.state.processed_files)} files to reconcile"
        )
        self.state.stage = PipelineStage.METADATA_RECONCILIATION
        
        total = len(self.state.processed_files)
        discrepancies_found = 0
        discrepancies_resolved = 0
        auto_resolved = 0
        camera_prompts = 0
        
        is_interactive = self.interaction_manager.mode in (
            InteractionMode.INTERACTIVE, InteractionMode.STEP_BY_STEP,
        )
        
        # Cache of subdirectory-level resolutions.
        # Key: (directory, field_name, frozenset(source_keys))
        # Value: chosen source key (str) | None (empty) | ('custom', value)
        subdir_resolutions: Dict[tuple, Any] = {}
        
        # Cache for camera-number subdirectory decisions.
        # Key: directory
        # Value: 'use' | 'discard' | ('custom', value)
        subdir_camera_decisions: Dict[str, Any] = {}
        
        # --- 1. Source-discrepancy resolution ---
        for i, (path_str, pf_data) in enumerate(self.state.processed_files.items()):
            path = Path(path_str)
            dir_path = str(path.parent)
            msrc = pf_data.get('metadata_sources', {})
            
            for field_name in self.RECONCILABLE_FIELDS:
                source_values = msrc.get(field_name, {})
                if len(source_values) < 2:
                    continue  # 0 or 1 source → no conflict
                
                # Check if all sources agree
                unique_values = set(str(v) for v in source_values.values())
                if len(unique_values) <= 1:
                    continue  # All sources agree
                
                discrepancies_found += 1
                
                source_keys_frozen = frozenset(source_values.keys())
                cache_key = (dir_path, field_name, source_keys_frozen)
                
                # Check subdirectory-level cache first
                if cache_key in subdir_resolutions:
                    cached = subdir_resolutions[cache_key]
                    self._apply_cached_resolution(
                        pf_data, field_name, source_values, cached
                    )
                    discrepancies_resolved += 1
                    continue
                
                # --- Non-interactive: auto-resolve via priority ---
                if not is_interactive:
                    winner = self._auto_resolve_discrepancy(
                        field_name, source_values
                    )
                    pf_data[field_name] = source_values[winner]
                    auto_resolved += 1
                    discrepancies_resolved += 1
                    logger.debug(
                        f"Auto-resolved {field_name} for {path.name}: "
                        f"chose {winner} = {source_values[winner]}"
                    )
                    self.action_logger.info(
                        f"Auto-resolved {field_name} for {path.name}: "
                        f"{winner} → {source_values[winner]}  "
                        f"(other: {', '.join(f'{k}={v}' for k, v in source_values.items() if k != winner)})"
                    )
                    continue
                
                # --- Interactive: ask the user ---
                decision_req = create_source_discrepancy_decision(
                    file_path=path,
                    field_name=field_name,
                    values_by_source=source_values,
                    directory=dir_path,
                )
                result = self.interaction_manager.request_decision(decision_req)
                
                resolution = self._interpret_discrepancy_result(
                    result, source_values, list(source_values.keys())
                )
                
                # Apply to this file
                self._apply_cached_resolution(
                    pf_data, field_name, source_values, resolution
                )
                discrepancies_resolved += 1
                
                # If subdirectory-wide, cache the resolution
                if result.outcome == DecisionOutcome.APPLY_TO_SUBDIRECTORY:
                    subdir_resolutions[cache_key] = resolution
                    logger.info(
                        f"Cached subdirectory resolution for "
                        f"{Path(dir_path).name}/{field_name}: {resolution}"
                    )
                
                # If deferred, flag for review
                if result.outcome == DecisionOutcome.DEFER:
                    pf_data['needs_review'] = True
                    pf_data['review_reason'] = (
                        f"Discrepancia_{field_name}"
                    )
            
            self._report_progress("Metadata Reconciliation", i + 1, total)
        
        # --- 2. Camera-number flag resolution ---
        for path_str, pf_data in self.state.processed_files.items():
            cam_flags = pf_data.get('camera_number_flags')
            if not cam_flags:
                continue
            
            path = Path(path_str)
            dir_path = str(path.parent)
            
            for cam_entry in cam_flags:
                numeric_id = cam_entry['numeric_id']
                raw_match = cam_entry['raw_match']
                
                camera_prompts += 1
                
                # --- Non-interactive: auto-discard camera numbers ---
                if not is_interactive:
                    self._apply_camera_resolution(pf_data, numeric_id, 'discard')
                    logger.debug(
                        f"Auto-discarded camera number '{numeric_id}' "
                        f"(raw: '{raw_match}') for {path.name}"
                    )
                    continue
                
                # Check subdirectory cache
                if dir_path in subdir_camera_decisions:
                    cached = subdir_camera_decisions[dir_path]
                    self._apply_camera_resolution(pf_data, numeric_id, cached)
                    continue
                
                # --- Interactive: ask the user ---
                decision_req = create_camera_number_decision(
                    file_path=path,
                    numeric_id=numeric_id,
                    raw_match=raw_match,
                    directory=dir_path,
                )
                result = self.interaction_manager.request_decision(decision_req)
                
                cam_resolution = self._interpret_camera_result(result, numeric_id)
                self._apply_camera_resolution(pf_data, numeric_id, cam_resolution)
                
                # Cache if subdirectory-wide
                if result.outcome == DecisionOutcome.APPLY_TO_SUBDIRECTORY:
                    subdir_camera_decisions[dir_path] = cam_resolution
                    logger.info(
                        f"Cached subdirectory camera decision for "
                        f"{Path(dir_path).name}: {cam_resolution}"
                    )
                
                if result.outcome == DecisionOutcome.DEFER:
                    pf_data['needs_review'] = True
                    pf_data['review_reason'] = (
                        f"Numero_camara_{numeric_id}"
                    )
            
            # Clean up transient flag now that it has been processed
            pf_data.pop('camera_number_flags', None)
        
        self.state.stage = PipelineStage.HASHING
        self._save_state()
        
        self.action_logger.stage_end(
            "METADATA_RECONCILIATION",
            f"Discrepancies found: {discrepancies_found} | Resolved: {discrepancies_resolved} | "
            f"Auto-resolved: {auto_resolved} | Camera prompts: {camera_prompts}"
        )
        logger.info(
            f"Metadata reconciliation complete — "
            f"{discrepancies_found} discrepancies ({auto_resolved} auto-resolved), "
            f"{camera_prompts} camera flags"
        )
    
    # ------------------------------------------------------------------
    # Reconciliation helpers
    # ------------------------------------------------------------------
    
    @staticmethod
    def _interpret_discrepancy_result(
        result: DecisionResult,
        source_values: Dict[str, Any],
        source_keys: list,
    ) -> Any:
        """Convert a DecisionResult into a cacheable resolution token.
        
        Returns one of:
            - source_key (str): use that source's value
            - None: leave field empty
            - ('custom', value): use an explicit custom value
        """
        if result.outcome in (DecisionOutcome.ACCEPT, DecisionOutcome.APPLY_TO_SUBDIRECTORY):
            idx = result.selected_option if result.selected_option is not None else 0
            if idx < len(source_keys):
                return source_keys[idx]  # chosen source
            elif idx == len(source_keys):
                return None  # "Leave field empty"
            else:
                # "Enter a custom value" — fall through to custom
                pass
        
        if result.outcome in (DecisionOutcome.CUSTOM, DecisionOutcome.APPLY_TO_SUBDIRECTORY):
            if result.custom_value is not None:
                return ('custom', result.custom_value)
        
        if result.outcome == DecisionOutcome.SKIP:
            return None
        
        # DEFER — leave as-is (last-writer-wins already applied)
        return '__defer__'
    
    @staticmethod
    def _apply_cached_resolution(
        pf_data: dict,
        field_name: str,
        source_values: Dict[str, Any],
        resolution,
    ):
        """Apply a resolution token to *pf_data[field_name]*."""
        if resolution == '__defer__':
            return  # leave current value
        if resolution is None:
            pf_data[field_name] = None
        elif isinstance(resolution, tuple) and resolution[0] == 'custom':
            pf_data[field_name] = resolution[1]
        elif isinstance(resolution, str) and resolution in source_values:
            pf_data[field_name] = source_values[resolution]
        # else: unexpected — leave as-is
    
    @staticmethod
    def _interpret_camera_result(result: DecisionResult, numeric_id: str) -> Any:
        """Convert a camera-flag DecisionResult into a cacheable token.
        
        Returns:
            - 'use': keep the numeric_id as specimen_id
            - 'discard': leave specimen_id untouched (don't use camera number)
            - ('custom', value): use a custom specimen_id
        """
        if result.outcome in (DecisionOutcome.ACCEPT, DecisionOutcome.APPLY_TO_SUBDIRECTORY):
            idx = result.selected_option if result.selected_option is not None else 1
            if idx == 0:
                return 'use'
            elif idx == 1:
                return 'discard'
            elif idx == 2 and result.custom_value:
                return ('custom', result.custom_value)
        
        if result.outcome == DecisionOutcome.CUSTOM and result.custom_value:
            return ('custom', result.custom_value)
        
        # Default: discard
        return 'discard'
    
    @staticmethod
    def _apply_camera_resolution(pf_data: dict, numeric_id: str, resolution):
        """Apply a camera-flag resolution to *pf_data*."""
        if resolution == 'use':
            # Only set specimen_id if not already populated by a proper pattern
            if not pf_data.get('specimen_id'):
                pf_data['specimen_id'] = numeric_id
        elif resolution == 'discard':
            pass  # Do nothing — camera number is ignored
        elif isinstance(resolution, tuple) and resolution[0] == 'custom':
            pf_data['specimen_id'] = resolution[1]

    def _run_hashing(self):
        """Stage 5: Compute hashes for images."""
        logger.info("Stage 5: Computing hashes...")
        self.state.stage = PipelineStage.HASHING
        
        # Filter to only image files
        image_files = [
            (path_str, pf_data)
            for path_str, pf_data in self.state.processed_files.items()
            if pf_data.get('file_type') == FileType.IMAGE.value
        ]
        
        total = len(image_files)
        logger.info(f"Hashing {total} image files")
        self.action_logger.stage_start("HASHING", f"{total} image files to hash")
        
        exif_count = 0
        hash_errors = 0
        
        for i, (path_str, pf_data) in enumerate(image_files):
            path = Path(path_str)
            
            if not path.exists():
                self.action_logger.warning(f"File missing during hashing: {path}")
                continue
            
            try:
                # Compute hashes and register
                hash_result = self.hash_registry.add(path)
                
                pf_data['md5_hash'] = hash_result.md5_hash
                pf_data['phash'] = hash_result.phash
                
                self.action_logger.hash_computed(
                    path.name,
                    md5=hash_result.md5_hash,
                    phash=hash_result.phash
                )
                
                # Extract EXIF date if available (uses DateTimeOriginal)
                from .file_utils import MetadataExtractor
                metadata = MetadataExtractor.extract(path)
                if metadata and metadata.date_taken:
                    exif_count += 1
                    
                    exif_year = None
                    if metadata.datetime_original:
                        exif_year = metadata.datetime_original.year
                    
                    path_year = pf_data.get('campaign_year')
                    
                    # Check for discrepancy between EXIF year and path-extracted year
                    if exif_year and path_year:
                        if int(path_year) != exif_year:
                            logger.warning(
                                f"Year discrepancy for {path.name}: "
                                f"EXIF year={exif_year}, path year={path_year}"
                            )
                            self.action_logger.warning(
                                f"Year discrepancy: {path.name} — "
                                f"EXIF={exif_year} vs path={path_year}"
                            )
                    
                    # If no campaign_year from path, use EXIF year
                    if not path_year and exif_year:
                        pf_data['campaign_year'] = exif_year
                    
                    self.action_logger.exif_extracted(
                        path.name,
                        date_taken=metadata.date_taken,
                        campaign_year=exif_year
                    )
                
            except Exception as e:
                logger.warning(f"Failed to hash {path}: {e}")
                pf_data['error'] = str(e)
                hash_errors += 1
                self.action_logger.hash_error(path.name, str(e))
            
            self.state.files_hashed = i + 1
            self._report_progress("Hashing", i + 1, total)
            
            # Save state periodically
            if (i + 1) % 100 == 0:
                self._save_state()
        
        self.state.stage = PipelineStage.DEDUPLICATION
        self._save_state()
        
        self.action_logger.stage_end(
            "HASHING",
            f"Hashed: {total - hash_errors}/{total} | EXIF dates: {exif_count} | Errors: {hash_errors}"
        )
        logger.info(f"Hashed {total} files")
    
    def _run_deduplication(self):
        """Stage 6: Identify duplicates."""
        logger.info("Stage 6: Identifying duplicates...")
        self.action_logger.stage_start("DEDUPLICATION")
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
            
            # Log the duplicate group
            self.action_logger.duplicate_group(
                dup_type=group.duplicate_type.value,
                hash_value=group.hash_value,
                file_count=len(group.files),
                files=group.files
            )
            
            # --- Check for metadata discrepancies across duplicate files ---
            metadata_fields_to_check = [
                'campaign_year', 'specimen_id', 'collection_code',
                'macroclass', 'taxonomic_class', 'determination',
            ]
            discrepancies: Dict[str, list] = {}
            group_file_strs = [str(f) for f in group.files]
            
            for field_name in metadata_fields_to_check:
                values_seen: Dict[str, list] = {}  # value -> [file_paths]
                for fpath_str in group_file_strs:
                    pf = self.state.processed_files.get(fpath_str)
                    if pf:
                        val = pf.get(field_name)
                        val_str = str(val) if val is not None else None
                        if val_str not in values_seen:
                            values_seen[val_str] = []
                        values_seen[val_str].append(fpath_str)
                
                # Filter out None-only entries; a discrepancy is when there are
                # ≥2 distinct NON-None values (a real conflict).
                # When only 1 non-None value exists alongside None entries,
                # auto-fill the non-None value into files that lack it.
                non_none_vals = {v for v in values_seen if v is not None}
                if len(non_none_vals) == 1 and None in values_seen:
                    # Only one meaningful value — propagate it to files
                    # that have None, no user intervention needed.
                    winning_val_str = next(iter(non_none_vals))
                    for fpath_str in values_seen[None]:
                        pf = self.state.processed_files.get(fpath_str)
                        if pf:
                            pf[field_name] = winning_val_str
                    logger.info(
                        f"Duplicate group (hash={group.hash_value[:16]}): "
                        f"auto-filled '{field_name}' = '{winning_val_str}' for "
                        f"{len(values_seen[None])} file(s) missing the value"
                    )
                    self.action_logger.info(
                        f"Duplicate auto-fill: {field_name}='{winning_val_str}' "
                        f"for {len(values_seen[None])} file(s) in group "
                        f"hash={group.hash_value[:16]}"
                    )
                elif len(non_none_vals) >= 2:
                    # Real discrepancy: ≥2 distinct non-None values
                    disc_list = []
                    for val_str, fpaths in values_seen.items():
                        display_val = val_str if val_str is not None else "(empty)"
                        for fp in fpaths:
                            disc_list.append((fp, display_val))
                    discrepancies[field_name] = disc_list
            
            if discrepancies:
                logger.warning(
                    f"Metadata discrepancy in duplicate group "
                    f"(hash={group.hash_value[:16]}): "
                    f"fields={list(discrepancies.keys())}"
                )
                self.action_logger.warning(
                    f"Duplicate metadata discrepancy: "
                    f"hash={group.hash_value[:16]} | "
                    f"fields={list(discrepancies.keys())} | "
                    f"files={[Path(f).name for f in group_file_strs]}"
                )
                
                # Ask user to resolve
                decision_request = create_duplicate_metadata_decision(
                    file_paths=[Path(f) for f in group_file_strs],
                    discrepancies=discrepancies,
                )
                result = self.interaction_manager.request_decision(decision_request)
                
                # Build a summary of discrepant fields for review_reason
                disc_fields_str = ', '.join(discrepancies.keys())
                
                # Apply resolution
                if result.outcome == DecisionOutcome.ACCEPT and result.selected_option is not None:
                    if result.selected_option < len(group_file_strs):
                        # Use metadata from the chosen file — apply ONLY to the
                        # original (kept) file.  Duplicate files keep their own
                        # metadata so the registry preserves both sets of info.
                        chosen_path = group_file_strs[result.selected_option]
                        chosen_pf = self.state.processed_files.get(chosen_path, {})
                        for field_name in discrepancies:
                            chosen_val = chosen_pf.get(field_name)
                            self.state.processed_files[original_path][field_name] = chosen_val
                        logger.info(
                            f"Resolved duplicate metadata: using values from "
                            f"{Path(chosen_path).name} (applied to original only)"
                        )
                    # else: last option = custom, handled below
                
                if result.outcome == DecisionOutcome.CUSTOM and result.custom_value:
                    # Parse custom values — expect "field=value" pairs
                    # Apply only to the original file
                    for pair in result.custom_value.split(";"):
                        pair = pair.strip()
                        if "=" in pair:
                            key, val = pair.split("=", 1)
                            key, val = key.strip(), val.strip()
                            if key in discrepancies:
                                self.state.processed_files[original_path][key] = val
                    logger.info("Resolved duplicate metadata with custom values (applied to original only)")
                
                # Routing depends on the user's decision:
                # - ACCEPT / CUSTOM: user actively resolved → duplicates go
                #   to Duplicados (normal flow).  No needs_review flag.
                # - DEFER: user deferred → ALL files in the group go to
                #   Revision_Manual so the user can decide later.
                if result.outcome == DecisionOutcome.DEFER:
                    short_hash = group.hash_value[:8]
                    for fpath_str in group_file_strs:
                        self.state.processed_files[fpath_str]['needs_review'] = True
                        self.state.processed_files[fpath_str]['review_reason'] = (
                            f"Discrepancia_metadata_{short_hash}"
                        )
            
            for dup_path in group.files[1:]:
                path_str = str(dup_path)
                if path_str in self.state.processed_files:
                    self.state.processed_files[path_str]['is_duplicate'] = True
                    self.state.processed_files[path_str]['duplicate_of'] = original_path
                    self.action_logger.duplicate_found(
                        Path(path_str), Path(original_path), group.hash_value
                    )
            
            # Store group info
            self.state.duplicate_groups.append({
                'type': group.duplicate_type.value,
                'hash': group.hash_value,
                'files': [str(f) for f in group.files],
            })
        
        self.state.stage = PipelineStage.ORGANIZING  # Move files first
        self._save_state()
        
        dup_count = sum(1 for pf in self.state.processed_files.values() if pf.get('is_duplicate'))
        self.action_logger.stage_end(
            "DEDUPLICATION",
            f"Groups: {len(all_groups)} | Duplicate files: {dup_count} | Unique: {len(self.state.processed_files) - dup_count}"
        )
        logger.info(f"Found {len(all_groups)} duplicate groups ({dup_count} duplicate files)")
    
    # ------------------------------------------------------------------
    # JPEG conversion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_as_jpeg(source_path: Path, dest_path: Path) -> bool:
        """Try to convert *source_path* to JPEG and save at *dest_path*.

        Returns True if the conversion succeeded, False if PIL could not
        open / convert the file (caller should fall back to a raw copy).
        """
        from .config import PIL_CONVERTIBLE_EXTENSIONS, JPEG_QUALITY

        if source_path.suffix.lower() not in PIL_CONVERTIBLE_EXTENSIONS:
            return False

        try:
            from PIL import Image
        except ImportError:
            return False

        try:
            with Image.open(source_path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')
                img.save(dest_path, format='JPEG', quality=JPEG_QUALITY)
            # Preserve source file timestamps as much as possible
            import os
            stat = os.stat(source_path)
            os.utime(dest_path, (stat.st_atime, stat.st_mtime))
            return True
        except Exception as e:
            logger.debug(
                f"PIL could not convert {source_path.name} to JPEG, "
                f"falling back to raw copy: {e}"
            )
            return False

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
        # Images are normalised to .jpg on output; keep original
        # extension only when PIL cannot convert the format.
        extension = original_path.suffix.lower()
        if pf_data.get('file_type') == 'image':
            from .config import PIL_CONVERTIBLE_EXTENSIONS
            if extension in PIL_CONVERTIBLE_EXTENSIONS:
                extension = '.jpg'
        new_filename = f"{year_str}_{specimen_id}_{file_uuid}{extension}"
        
        return new_filename
    
    def _get_image_destination(self, pf_data: Dict[str, Any], source_path: Path) -> Tuple[Path, str]:
        """
        Determine destination path for an image file.
        
        Structure (Las Hoyas / default): Campaña_[AÑO]/Macroclase/Clase/Genero/filename
        Structure (other collections): Otras_Colecciones/<Collection>/Campaña_[AÑO]/...
        
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
        # Handle items needing review FIRST — this takes priority over
        # is_duplicate because discrepant duplicates need human attention.
        if pf_data.get('needs_review'):
            reason = pf_data.get('review_reason', 'Otro')
            # Sanitize reason for use as directory name (remove Windows-invalid chars)
            import re as _re
            reason = _re.sub(r'[<>:"/\\|?*\[\]\']', '', reason).strip()
            if not reason:
                reason = 'Otro'
            dest_dir = self.output_dir / 'Revision_Manual' / reason
            new_filename = self._generate_new_filename(pf_data, source_path)
            return dest_dir, new_filename
        
        # Handle duplicates (without review flags)
        if pf_data.get('is_duplicate'):
            dest_dir = self.output_dir / 'Duplicados'
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
        # Las Hoyas is the default collection — goes to root.
        # Buenache/Montsec go under Otras_Colecciones/<collection_name>/
        collection = pf_data.get('collection_code', '')
        if collection in ('BUE',):
            dest_dir = self.output_dir / 'Otras_Colecciones' / 'Buenache'
        elif collection in ('MON',):
            dest_dir = self.output_dir / 'Otras_Colecciones' / 'Montsec'
        else:
            # LH, unknown, or anything else → root (Las Hoyas is default)
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
        
        if macroclass:
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
        
        # Determine collection prefix
        # Las Hoyas is default (root), Buenache/Montsec under Otras_Colecciones/
        collection = pf_data.get('collection_code', '')
        if collection in ('BUE',):
            collection_base = self.output_dir / 'Otras_Colecciones' / 'Buenache'
        elif collection in ('MON',):
            collection_base = self.output_dir / 'Otras_Colecciones' / 'Montsec'
        else:
            collection_base = self.output_dir
        
        # Determine base directory
        if is_text:
            dest_dir = collection_base / 'Archivos_Texto'
        else:
            dest_dir = collection_base / 'Otros_Archivos'
        
        # Generate new filename: <uuid>_<original_name>
        original_name = source_path.name
        # Sanitize original name
        safe_name = re.sub(r'[^\w\.\-]', '_', original_name)
        new_filename = f"{file_uuid}_{safe_name}"
        
        return dest_dir, new_filename
    
    def _run_organizing(self):
        """Stage 7: Organize files into collection folders with renaming.
        
        NOTE: This runs BEFORE registry generation, so current_path in the 
        registry reflects the final destination of each file.
        """
        logger.info("Stage 7: Organizing files (moving to final destinations)...")
        self.action_logger.stage_start("ORGANIZING", f"dry_run={self.dry_run}")
        self.state.stage = PipelineStage.ORGANIZING
        
        if self.dry_run:
            logger.info("DRY RUN - Not moving files")
            self.action_logger.info("DRY RUN mode — files will not be moved")
        
        # Create base output directories
        # Las Hoyas is default (root), other collections under Otras_Colecciones/
        base_dirs = [
            self.output_dir / 'Otras_Colecciones' / 'Buenache',
            self.output_dir / 'Otras_Colecciones' / 'Montsec',
            self.output_dir / 'Duplicados',
            self.output_dir / 'Revision_Manual',
            self.output_dir / 'Archivos_Texto',
            self.output_dir / 'Otros_Archivos',
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
            
            # Log destination decision with reasoning
            reason_parts = []
            if pf_data.get('is_duplicate'):
                reason_parts.append("duplicate")
            elif pf_data.get('needs_review'):
                reason_parts.append(f"review:{pf_data.get('review_reason', '?')}")
            else:
                reason_parts.append(f"collection={pf_data.get('collection_code', 'default')}")
                reason_parts.append(f"class={pf_data.get('taxonomic_class', 'unknown')}")
                reason_parts.append(f"year={pf_data.get('campaign_year', 'unknown')}")
            self.action_logger.file_destination(
                source_path, dest_path,
                reason=" | ".join(reason_parts)
            )
            
            # Move/copy file (convert images to .jpg when possible)
            if not self.dry_run:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    converted = False
                    if file_type == FileType.IMAGE:
                        converted = self._copy_as_jpeg(source_path, dest_path)
                    if not converted:
                        shutil.copy2(source_path, dest_path)
                    pf_data['moved'] = True
                    
                    # Log the action
                    self.action_logger.file_copied(source_path, dest_path)
                except Exception as e:
                    logger.warning(f"Failed to copy {source_path}: {e}")
                    self.action_logger.error(f"Failed to copy {source_path}", e)
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
        
        self.state.stage = PipelineStage.REGISTRY_GENERATION  # Registry now runs AFTER organizing
        self._save_state()
        
        self.action_logger.stage_end(
            "ORGANIZING",
            f"Organized: {self.state.files_organized} files | dry_run={self.dry_run}"
        )
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
        """Stage 8: Generate Excel registries for images, text files, and other files.
        
        IMPORTANT: This runs AFTER organizing (file move) so that current_path
        reflects the final destination of each file.
        Duplicates are NOT included in anotaciones.xlsx — they get their own
        registry file inside the Duplicados folder.
        """
        logger.info("Stage 8: Generating registries (logging to Excel AFTER file move)...")
        self.action_logger.stage_start("REGISTRY_GENERATION")
        self.state.stage = PipelineStage.REGISTRY_GENERATION
        
        # Create Excel manager
        registries_dir = self.output_dir / "registries"
        registries_dir.mkdir(parents=True, exist_ok=True)
        excel_manager = ExcelManager(registries_dir)
        
        # Separate Excel manager for duplicates registry
        duplicados_dir = self.output_dir / "Duplicados"
        duplicados_dir.mkdir(parents=True, exist_ok=True)
        
        # Counters for different file types
        image_count = 0
        duplicate_count = 0
        text_count = 0
        other_count = 0
        
        # Collect duplicate records separately
        duplicate_records: list = []
        
        # Add processed files to appropriate registries
        for path_str, pf_data in self.state.processed_files.items():
            file_type_str = pf_data.get('file_type')
            file_type = FileType(file_type_str) if file_type_str else FileType.OTHER
            
            if file_type == FileType.IMAGE:
                # --- Hashes registry: ALL images (including duplicates) ---
                file_uuid = pf_data.get('file_uuid') or ImageRecord.generate_uuid()
                md5 = pf_data.get('md5_hash', '')
                phash = pf_data.get('phash', '')
                excel_manager.add_hash(
                    uuid=file_uuid,
                    md5_hash=md5,
                    phash=phash,
                    file_path=Path(pf_data.get('destination_path') or path_str),
                )
                
                # --- Check if duplicate ---
                if pf_data.get('is_duplicate'):
                    # Build comments for duplicate record
                    comments = [f"[AUTO] duplicate_of={pf_data.get('duplicate_of')}"]
                    if pf_data.get('needs_review'):
                        comments.append(f"[REVIEW] {pf_data.get('review_reason')}")
                    
                    dup_record = ImageRecord(
                        uuid=file_uuid,
                        specimen_id=pf_data.get('specimen_id'),
                        original_path=path_str,
                        current_path=pf_data.get('destination_path'),
                        macroclass_label=pf_data.get('macroclass') or get_macroclass_folder(pf_data.get('taxonomic_class')),
                        class_label=pf_data.get('taxonomic_class'),
                        genera_label=pf_data.get('determination'),
                        campaign_year=str(pf_data.get('campaign_year')) if pf_data.get('campaign_year') else None,
                        fuente=pf_data.get('collection_code'),
                        comentarios=" | ".join(comments),
                    )
                    duplicate_records.append(dup_record)
                    duplicate_count += 1
                    self.action_logger.registry_entry(
                        "duplicate", pf_data.get('filename', ''),
                        specimen_id=pf_data.get('specimen_id')
                    )
                    continue  # Do NOT add duplicates to main anotaciones.xlsx
                
                # --- Main anotaciones.xlsx: non-duplicate images only ---
                comments = []
                if pf_data.get('needs_review'):
                    comments.append(f"[REVIEW] {pf_data.get('review_reason')}")
                if pf_data.get('collection_code'):
                    comments.append(f"[AUTO] collection={pf_data.get('collection_code')}")
                if pf_data.get('specimen_id'):
                    comments.append(f"[AUTO] specimen_id extracted from path/filename")
                
                # Accumulate original_paths: if this file has duplicates,
                # collect all their original_paths into a ';'-separated list.
                all_orig_paths = [path_str]
                for dup_info in self.state.processed_files.values():
                    if dup_info.get('duplicate_of') == path_str:
                        dup_orig = str(dup_info.get('original_path', ''))
                        if dup_orig and dup_orig not in all_orig_paths:
                            all_orig_paths.append(dup_orig)
                combined_original_path = '; '.join(all_orig_paths)
                
                # Get macroclass — None if unknown (no more Unsorted_Macroclass)
                macroclass = pf_data.get('macroclass') or get_macroclass_folder(pf_data.get('taxonomic_class'))
                
                record = ImageRecord(
                    uuid=file_uuid,
                    specimen_id=pf_data.get('specimen_id'),
                    original_path=combined_original_path,
                    current_path=pf_data.get('destination_path'),
                    macroclass_label=macroclass,
                    class_label=pf_data.get('taxonomic_class'),
                    genera_label=pf_data.get('determination'),
                    campaign_year=str(pf_data.get('campaign_year')) if pf_data.get('campaign_year') else None,
                    fuente=pf_data.get('collection_code'),
                    comentarios=" | ".join(comments) if comments else "[AUTO] processed by pipeline",
                )
                excel_manager.add_image(record)
                image_count += 1
                self.action_logger.registry_entry(
                    "image", pf_data.get('filename', ''),
                    specimen_id=pf_data.get('specimen_id')
                )
                
            elif file_type == FileType.TEXT:
                original_path = Path(path_str)
                current_path = Path(pf_data.get('destination_path') or path_str)
                excel_manager.add_text_file(original_path, current_path)
                text_count += 1
                self.action_logger.registry_entry("text", pf_data.get('filename', ''))
                
            else:
                original_path = Path(path_str)
                current_path = Path(pf_data.get('destination_path') or path_str)
                excel_manager.add_other_file(original_path, current_path)
                other_count += 1
                self.action_logger.registry_entry("other", pf_data.get('filename', ''))
        
        # Save all main registries
        excel_manager.save_all()
        logger.info(f"Saved registries to: {registries_dir}")
        logger.info(f"  Images: {image_count}, Text files: {text_count}, Other: {other_count}")
        
        # Save duplicate registry to Duplicados/duplicados_registro.xlsx
        if duplicate_records:
            import pandas as pd
            from dataclasses import asdict
            dup_df = pd.DataFrame([asdict(r) for r in duplicate_records])
            dup_registry_path = duplicados_dir / "duplicados_registro.xlsx"
            dup_df.to_excel(dup_registry_path, index=False)
            logger.info(f"Saved {duplicate_count} duplicate records to: {dup_registry_path}")
        
        # Also save a summary CSV for easy viewing
        summary_path = self.output_dir / "processing_summary.csv"
        self._save_summary_csv(summary_path)
        logger.info(f"Saved summary: {summary_path}")
        
        # Transition to COMPLETED
        self.state.stage = PipelineStage.COMPLETED
        self._save_state()
        
        self.action_logger.stage_end(
            "REGISTRY_GENERATION",
            f"Images: {image_count} | Duplicates: {duplicate_count} | Text: {text_count} | Other: {other_count}"
        )
        logger.info("Registry generation complete")
    
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
    progress_callback: Callable = None,
    use_staging: bool = True
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
        use_staging: If True, output to staging dir for validation before merge
        
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
        progress_callback=progress_callback,
        use_staging=use_staging,
    )
    
    return orchestrator.run(resume=resume)
