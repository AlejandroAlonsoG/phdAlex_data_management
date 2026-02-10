"""
Logging module for the data ordering tool.
Provides structured logging to both console and file.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from enum import Enum


class LogAction(Enum):
    """Types of actions that can be logged."""
    # Pipeline stage transitions
    STAGE_START = "STAGE_START"
    STAGE_END = "STAGE_END"
    
    # Directory operations
    DIR_ENTER = "DIR_ENTER"
    DIR_EMPTY_DELETED = "DIR_EMPTY_DELETED"
    DIR_CREATED = "DIR_CREATED"
    
    # File operations
    FILE_SCANNED = "FILE_SCANNED"
    FILE_MOVED = "FILE_MOVED"
    FILE_COPIED = "FILE_COPIED"
    FILE_DELETED = "FILE_DELETED"
    FILE_RENAMED = "FILE_RENAMED"
    FILE_CLASSIFIED = "FILE_CLASSIFIED"
    FILE_DESTINATION = "FILE_DESTINATION"
    
    # Extraction operations
    METADATA_EXTRACTED = "METADATA_EXTRACTED"
    SPECIMEN_ID_FOUND = "SPECIMEN_ID_FOUND"
    SPECIMEN_ID_MISSING = "SPECIMEN_ID_MISSING"
    TAXONOMY_FOUND = "TAXONOMY_FOUND"
    COLLECTION_DETECTED = "COLLECTION_DETECTED"
    DATE_FOUND = "DATE_FOUND"
    EXIF_EXTRACTED = "EXIF_EXTRACTED"
    
    # LLM operations
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_RATE_LIMIT_WAIT = "LLM_RATE_LIMIT_WAIT"
    LLM_PATH_ANALYSIS = "LLM_PATH_ANALYSIS"
    LLM_REGEX_GENERATED = "LLM_REGEX_GENERATED"
    LLM_REGEX_APPLIED = "LLM_REGEX_APPLIED"
    LLM_PER_FILE = "LLM_PER_FILE"
    LLM_ERROR = "LLM_ERROR"
    
    # Hashing
    HASH_COMPUTED = "HASH_COMPUTED"
    HASH_ERROR = "HASH_ERROR"
    
    # Deduplication
    DUPLICATE_FOUND = "DUPLICATE_FOUND"
    DUPLICATE_GROUP = "DUPLICATE_GROUP"
    DUPLICATE_MERGED = "DUPLICATE_MERGED"
    
    # Registry
    REGISTRY_ENTRY = "REGISTRY_ENTRY"
    
    # Manual review
    MANUAL_REVIEW_QUEUED = "MANUAL_REVIEW_QUEUED"
    MANUAL_REVIEW_RESOLVED = "MANUAL_REVIEW_RESOLVED"
    
    # Errors and warnings
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class DataOrderingLogger:
    """
    Logger for the data ordering process.
    Logs to both console and a timestamped file.
    """
    
    def __init__(self, log_dir: Path, session_name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory where log files will be stored
            session_name: Optional name for this session (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or timestamp
        self.log_file = self.log_dir / f"session_{self.session_name}.log"
        
        # Setup logging
        self.logger = logging.getLogger(f"data_ordering_{self.session_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler - detailed
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Console handler - less verbose
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)-8s | %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Log session start
        self.log(LogAction.INFO, f"Session started: {self.session_name}")
        self.log(LogAction.INFO, f"Log file: {self.log_file}")
    
    def log(self, action: LogAction, message: str, **kwargs):
        """
        Log an action with optional extra data.
        
        Args:
            action: The type of action being logged
            message: Human-readable message
            **kwargs: Additional data to include in the log
        """
        # Format extra data
        extra_str = ""
        if kwargs:
            extra_parts = [f"{k}={v}" for k, v in kwargs.items()]
            extra_str = " | " + " | ".join(extra_parts)
        
        full_message = f"[{action.value}] {message}{extra_str}"
        
        # Route to appropriate log level
        if action == LogAction.ERROR:
            self.logger.error(full_message)
        elif action == LogAction.WARNING:
            self.logger.warning(full_message)
        elif action in [LogAction.LLM_RATE_LIMIT_WAIT]:
            # Rate limit waits are debug-only (noisy)
            self.logger.debug(full_message)
        else:
            # Everything else at INFO so it appears in the log file
            self.logger.info(full_message)
    
    def dir_enter(self, path: Path):
        """Log entering a directory."""
        self.log(LogAction.DIR_ENTER, f"Entering directory: {path}")
    
    def dir_empty_deleted(self, path: Path):
        """Log deletion of empty directory."""
        self.log(LogAction.DIR_EMPTY_DELETED, f"Deleted empty directory: {path}")
    
    def file_moved(self, source: Path, dest: Path, reason: str = ""):
        """Log file movement."""
        self.log(LogAction.FILE_MOVED, f"{source} -> {dest}", reason=reason)
    
    def file_copied(self, source: Path, dest: Path):
        """Log file copy."""
        self.log(LogAction.FILE_COPIED, f"{source} -> {dest}")
    
    def specimen_id_found(self, filename: str, specimen_id: str, source: str = "regex"):
        """Log specimen ID extraction."""
        self.log(LogAction.SPECIMEN_ID_FOUND, f"Found ID: {specimen_id}", 
                 filename=filename, source=source)
    
    def taxonomy_found(self, path_or_file: str, level: str, value: str):
        """Log taxonomy extraction."""
        self.log(LogAction.TAXONOMY_FOUND, f"{level}: {value}", source=path_or_file)
    
    def duplicate_found(self, file1: Path, file2: Path, hash_value: str):
        """Log duplicate detection."""
        self.log(LogAction.DUPLICATE_FOUND, f"Duplicate detected", 
                 file1=str(file1), file2=str(file2), hash=hash_value[:16])
    
    def duplicate_group(self, dup_type: str, hash_value: str, file_count: int, files: list):
        """Log a duplicate group."""
        file_list = ", ".join(str(f) for f in files[:5])
        if len(files) > 5:
            file_list += f" ... +{len(files)-5} more"
        self.log(LogAction.DUPLICATE_GROUP, f"Group ({dup_type}): {file_count} files",
                 hash=hash_value[:16], files=file_list)
    
    def llm_request(self, prompt_summary: str):
        """Log LLM request."""
        self.log(LogAction.LLM_REQUEST, prompt_summary[:200])
    
    def llm_response(self, response_summary: str):
        """Log LLM response."""
        self.log(LogAction.LLM_RESPONSE, response_summary[:300])
    
    def llm_path_analysis(self, directory: str, macroclass: str, taxonomic_class: str,
                          genus: str, collection: str, campaign_year, confidence: float):
        """Log the result of an LLM path analysis."""
        self.log(LogAction.LLM_PATH_ANALYSIS, f"Path analysis: {directory}",
                 macroclass=macroclass, taxonomic_class=taxonomic_class,
                 genus=genus, collection=collection, campaign_year=campaign_year,
                 confidence=f"{confidence:.1%}")
    
    def llm_regex_generated(self, directory: str, regex: str, extractable_fields: list):
        """Log a regex generated by LLM."""
        self.log(LogAction.LLM_REGEX_GENERATED, f"Regex for {directory}",
                 regex=regex, fields=extractable_fields)
    
    def llm_regex_applied(self, directory: str, total: int, matched: int, failed: int):
        """Log regex application results."""
        rate = f"{matched/total:.1%}" if total > 0 else "N/A"
        self.log(LogAction.LLM_REGEX_APPLIED, f"Regex applied in {directory}",
                 total=total, matched=matched, failed=failed, success_rate=rate)
    
    def llm_per_file(self, filename: str, specimen_id: str, result_summary: str):
        """Log a per-file LLM fallback analysis."""
        self.log(LogAction.LLM_PER_FILE, f"Per-file analysis: {filename}",
                 specimen_id=specimen_id or "None", result=result_summary)
    
    def llm_error(self, stage: str, directory_or_file: str, error: str):
        """Log an LLM error."""
        self.log(LogAction.LLM_ERROR, f"LLM error in {stage}: {directory_or_file}",
                 error=error)
    
    def llm_rate_wait(self, seconds: float):
        """Log rate limit waiting."""
        self.log(LogAction.LLM_RATE_LIMIT_WAIT, f"Waiting {seconds:.1f}s for rate limit")
    
    def stage_start(self, stage_name: str, detail: str = ""):
        """Log the start of a pipeline stage."""
        self.log(LogAction.STAGE_START, f"=== STAGE START: {stage_name} === {detail}")
    
    def stage_end(self, stage_name: str, detail: str = ""):
        """Log the end of a pipeline stage."""
        self.log(LogAction.STAGE_END, f"=== STAGE END: {stage_name} === {detail}")
    
    def file_scanned(self, path: Path, file_type: str, file_size: int):
        """Log a file being discovered during scanning."""
        self.log(LogAction.FILE_SCANNED, f"Scanned: {path}",
                 file_type=file_type, size_bytes=file_size)
    
    def file_classified(self, path: Path, file_type: str):
        """Log file classification."""
        self.log(LogAction.FILE_CLASSIFIED, f"Classified: {path.name}", file_type=file_type)
    
    def file_destination(self, source: Path, dest: Path, reason: str = ""):
        """Log the destination decision for a file."""
        self.log(LogAction.FILE_DESTINATION, f"{source.name} -> {dest}",
                 reason=reason)
    
    def specimen_id_missing(self, filename: str, reason: str = ""):
        """Log when no specimen ID could be extracted."""
        self.log(LogAction.SPECIMEN_ID_MISSING, f"No specimen ID: {filename}",
                 reason=reason)
    
    def collection_detected(self, filename_or_path: str, collection: str, source: str = ""):
        """Log collection detection."""
        self.log(LogAction.COLLECTION_DETECTED, f"Collection: {collection}",
                 file=filename_or_path, source=source)
    
    def hash_computed(self, filename: str, md5: str, phash: str = None):
        """Log hash computation."""
        self.log(LogAction.HASH_COMPUTED, f"Hashed: {filename}",
                 md5=md5[:16] + "..." if md5 else "None",
                 phash=phash[:16] + "..." if phash else "None")
    
    def hash_error(self, filename: str, error: str):
        """Log hash computation failure."""
        self.log(LogAction.HASH_ERROR, f"Hash failed: {filename}", error=error)
    
    def exif_extracted(self, filename: str, date_taken: str = None, 
                       camera: str = None, campaign_year: int = None):
        """Log EXIF metadata extraction."""
        self.log(LogAction.EXIF_EXTRACTED, f"EXIF: {filename}",
                 date_taken=date_taken or "None",
                 campaign_year=campaign_year or "None")
    
    def metadata_extracted(self, filename: str, specimen_id: str = None,
                           collection: str = None, taxonomy: str = None,
                           campaign_year=None, source: str = ""):
        """Log extracted metadata summary for a file."""
        self.log(LogAction.METADATA_EXTRACTED, f"Metadata: {filename}",
                 specimen_id=specimen_id or "None",
                 collection=collection or "None",
                 taxonomy=taxonomy or "None",
                 campaign_year=campaign_year or "None",
                 source=source)
    
    def registry_entry(self, file_type: str, filename: str, specimen_id: str = None):
        """Log a registry entry being created."""
        self.log(LogAction.REGISTRY_ENTRY, f"Registry ({file_type}): {filename}",
                 specimen_id=specimen_id or "None")
    
    def manual_review_queued(self, item_type: str, item_id: str, reason: str):
        """Log item queued for manual review."""
        self.log(LogAction.MANUAL_REVIEW_QUEUED, reason, 
                 type=item_type, id=item_id)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log an error."""
        if exception:
            self.log(LogAction.ERROR, f"{message}: {type(exception).__name__}: {exception}")
        else:
            self.log(LogAction.ERROR, message)
    
    def warning(self, message: str):
        """Log a warning."""
        self.log(LogAction.WARNING, message)
    
    def info(self, message: str):
        """Log info message."""
        self.log(LogAction.INFO, message)
    
    def close(self):
        """Close the logger and finalize the session."""
        self.log(LogAction.INFO, "Session ended")
        for handler in self.logger.handlers:
            handler.close()


# Quick test function
def test_logger():
    """Quick test of the logger module."""
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DataOrderingLogger(log_dir, "test_session")
        
        # Test various log methods
        logger.info("Starting test")
        logger.dir_enter(Path("/some/directory"))
        logger.specimen_id_found("MUPA-12345678-a.jpg", "12345678", "regex")
        logger.taxonomy_found("/path/Arthropoda/Insecta", "class", "Insecta")
        logger.warning("This is a warning")
        logger.error("This is an error", ValueError("test error"))
        logger.close()
        
        # Verify log file exists and has content
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) == 1, "Should create one log file"
        
        content = log_files[0].read_text(encoding='utf-8')
        assert "Starting test" in content
        assert "12345678" in content
        assert "Insecta" in content
        
        print("✓ Logger test passed!")
        print(f"Log file content preview:\n{content[:500]}...")


if __name__ == "__main__":
    test_logger()
