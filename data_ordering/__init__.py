"""
Data Ordering Tool
==================

A tool for organizing and preparing fossil image datasets for deep learning.

Modules:
- config: Configuration settings
- logger_module: Structured logging
- excel_manager: Registry file management (Excel/CSV)
- file_utils: File system operations
- pattern_extractor: Specimen ID, date, and taxonomy pattern extraction
- image_hasher: Perceptual and MD5 hashing for duplicate detection
- file_scanner: File system scanning and categorization
- llm_integration: LLM-based directory analysis (Gemini)
- main_orchestrator: Main pipeline orchestrator
"""

from .config import (
    Config, config, COLLECTIONS, CollectionConfig, detect_collection,
    MACROCLASSES, CLASS_TO_MACROCLASS, get_macroclass, get_macroclass_folder
)
from .logger_module import DataOrderingLogger, LogAction
from .excel_manager import ExcelManager, ImageRecord, TextFileRecord, OtherFileRecord
from .file_utils import (
    FileClassifier, DirectoryWalker, MetadataExtractor, 
    FileOperations, FileType, FileInfo
)
from .pattern_extractor import (
    PatternExtractor, SpecimenIdExtractor, DateExtractor, PathAnalyzer,
    ExtractionResult, SpecimenIdMatch, DateMatch, TaxonomyHint
)
from .image_hasher import (
    ImageHasher, HashRegistry, ImageHash, DuplicateType, DuplicateGroup,
    hash_image, find_duplicates_in_folder
)
from .file_scanner import (
    FileScanner, ScannedFile, ScanProgress, ScanStatus,
    scan_directory
)
from .llm_integration import (
    GeminiClient, GitHubModelsClient, DirectoryAnalyzer, DirectoryAnalysis, RateLimiter,
    MockGeminiClient, analyze_directory, get_llm_client,
    GENAI_AVAILABLE, OPENAI_AVAILABLE
)
from .interaction_manager import (
    InteractionManager, InteractionMode, DecisionType, DecisionOutcome,
    DecisionRequest, DecisionResult
)
from .main_orchestrator import (
    PipelineOrchestrator, PipelineState, ProcessedFile, PipelineStage,
    run_pipeline
)
from .merge_output import OutputMerger

__version__ = "0.9.0"
__all__ = [
    'Config', 'config', 'COLLECTIONS', 'CollectionConfig', 'detect_collection',
    'MACROCLASSES', 'CLASS_TO_MACROCLASS', 'get_macroclass', 'get_macroclass_folder',
    'DataOrderingLogger', 'LogAction',
    'ExcelManager', 'ImageRecord', 'TextFileRecord', 'OtherFileRecord',
    'FileClassifier', 'DirectoryWalker', 'MetadataExtractor',
    'FileOperations', 'FileType', 'FileInfo',
    'PatternExtractor', 'SpecimenIdExtractor', 'DateExtractor', 'PathAnalyzer',
    'ExtractionResult', 'SpecimenIdMatch', 'DateMatch', 'TaxonomyHint',
    'ImageHasher', 'HashRegistry', 'ImageHash', 'DuplicateType', 'DuplicateGroup',
    'hash_image', 'find_duplicates_in_folder',
    'FileScanner', 'ScannedFile', 'ScanProgress', 'ScanStatus', 'scan_directory',
    'GeminiClient', 'GitHubModelsClient', 'DirectoryAnalyzer', 'DirectoryAnalysis', 
    'RateLimiter', 'MockGeminiClient', 'analyze_directory', 'get_llm_client',
    'GENAI_AVAILABLE', 'OPENAI_AVAILABLE',
    'InteractionManager', 'InteractionMode', 'DecisionType', 'DecisionOutcome',
    'DecisionRequest', 'DecisionResult',
    'PipelineOrchestrator', 'PipelineState', 'ProcessedFile', 'PipelineStage',
    'run_pipeline',
    'OutputMerger',
]
