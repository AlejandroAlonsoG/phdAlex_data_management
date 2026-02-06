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
    # Directory operations
    DIR_ENTER = "DIR_ENTER"
    DIR_EMPTY_DELETED = "DIR_EMPTY_DELETED"
    DIR_CREATED = "DIR_CREATED"
    
    # File operations
    FILE_MOVED = "FILE_MOVED"
    FILE_COPIED = "FILE_COPIED"
    FILE_DELETED = "FILE_DELETED"
    FILE_RENAMED = "FILE_RENAMED"
    FILE_CLASSIFIED = "FILE_CLASSIFIED"
    
    # Extraction operations
    METADATA_EXTRACTED = "METADATA_EXTRACTED"
    SPECIMEN_ID_FOUND = "SPECIMEN_ID_FOUND"
    TAXONOMY_FOUND = "TAXONOMY_FOUND"
    DATE_FOUND = "DATE_FOUND"
    
    # LLM operations
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    LLM_RATE_LIMIT_WAIT = "LLM_RATE_LIMIT_WAIT"
    
    # Deduplication
    DUPLICATE_FOUND = "DUPLICATE_FOUND"
    DUPLICATE_MERGED = "DUPLICATE_MERGED"
    
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
        elif action in [LogAction.LLM_REQUEST, LogAction.LLM_RESPONSE, 
                        LogAction.METADATA_EXTRACTED, LogAction.LLM_RATE_LIMIT_WAIT]:
            self.logger.debug(full_message)
        else:
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
    
    def llm_request(self, prompt_summary: str):
        """Log LLM request."""
        self.log(LogAction.LLM_REQUEST, prompt_summary[:100])
    
    def llm_rate_wait(self, seconds: float):
        """Log rate limit waiting."""
        self.log(LogAction.LLM_RATE_LIMIT_WAIT, f"Waiting {seconds:.1f}s for rate limit")
    
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
