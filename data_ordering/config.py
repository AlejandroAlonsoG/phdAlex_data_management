"""
Configuration for the data ordering tool.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# === Macroclass Definitions ===
# Each macroclass groups multiple taxonomic classes for simplified organization
# Based on "Diseño base de datos" document

MACROCLASSES = {
    'Botany': {
        'name': 'Botany',
        'name_es': 'Plantas',
        'folder': 'Botany',
        'classes': frozenset({
            'Angiosperma', 'Bryophyta', 'Charophyceae', 'Equisetopsida',
            'Eudicotyledoneae', 'Marchantiopsida', 'Pinales', 'Pinopsida',
            'planta', 'Pteridophyta', 'Pterophyta'
        }),
    },
    'Arthropoda': {
        'name': 'Arthropoda',
        'name_es': 'Artrópodos',
        'folder': 'Arthropoda',
        'classes': frozenset({
            'Arachnida', 'Branchiopoda', 'Crustacea', 'Diplopoda',
            'Insecta', 'Malacostraca', 'Ostracoda'
        }),
    },
    'Mollusca_Vermes': {
        'name': 'Mollusca & Vermes',
        'name_es': 'Moluscos y gusanos',
        'folder': 'Mollusca_Vermes',
        'classes': frozenset({
            'Bivalvia', 'Clitellata', 'Gastropoda', 'Nematoda'
        }),
    },
    'Pisces': {
        'name': 'Pisces',
        'name_es': 'Peces',
        'folder': 'Pisces',
        'classes': frozenset({
            'Actinopterygii', 'Chondrichthyes', 'Osteichthyes', 'Sarcopterygii'
        }),
    },
    'Tetrapoda': {
        'name': 'Tetrapoda',
        'name_es': 'Tetrápodos',
        'folder': 'Tetrapoda',
        'classes': frozenset({
            'Amphibia', ' Amphibia', 'Aves', 'Sauropsida', ' Sauropsida',
            'Reptilia', 'Tetrapoda', 'Vertebrata'
        }),
    },
    'Ichnofossils': {
        'name': 'Ichnofossils',
        'name_es': 'Icnofósiles',
        'folder': 'Ichnofossils',
        'classes': frozenset({
            'icnofósil', 'icnofósiles', 'Icnofósil', 'Icnofósiles'
        }),
    },
}

# Build reverse lookup: class -> macroclass
CLASS_TO_MACROCLASS: Dict[str, str] = {}
for macro_key, macro_data in MACROCLASSES.items():
    for cls in macro_data['classes']:
        CLASS_TO_MACROCLASS[cls.lower().strip()] = macro_key


def get_macroclass(class_label: Optional[str]) -> Optional[str]:
    """
    Get the macroclass for a given taxonomic class.
    
    Args:
        class_label: The taxonomic class (e.g., 'Insecta', 'Osteichthyes')
        
    Returns:
        Macroclass key (e.g., 'Arthropoda', 'Pisces') or None if not found
    """
    if not class_label:
        return None
    return CLASS_TO_MACROCLASS.get(class_label.lower().strip())


def get_macroclass_folder(class_label: Optional[str]) -> Optional[str]:
    """
    Get the folder name for a given taxonomic class's macroclass.
    
    Args:
        class_label: The taxonomic class
        
    Returns:
        Folder name (e.g., 'Arthropoda') or None if not found
    """
    macro_key = get_macroclass(class_label)
    if macro_key and macro_key in MACROCLASSES:
        return MACROCLASSES[macro_key]['folder']
    return None

# === File extension sets ===
IMAGE_WITH_METADATA_SUPPORT = {
    '.jpg', '.jpeg', '.tiff', '.tif',
    '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raw'
}

IMAGE_WITHOUT_METADATA_SUPPORT = {
    '.png', '.gif', '.bmp', '.svg', '.ico', '.webp',
    '.psd', '.ai', '.eps'
}

IMAGE_EXTENSIONS = IMAGE_WITH_METADATA_SUPPORT | IMAGE_WITHOUT_METADATA_SUPPORT

# JPEG normalization settings
JPEG_QUALITY = 95  # Quality for JPEG conversion (1-100)

# Image extensions that PIL/Pillow can reliably open and convert to JPEG.
# Formats outside this set will keep their original format and use
# file-bytes MD5 instead of normalised-JPEG MD5.
PIL_CONVERTIBLE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
    '.webp', '.psd',  # Pillow reads flattened PSD
}

TEXT_EXTENSIONS = {
    '.txt', '.pdf', '.doc', '.docx', '.odt', '.rtf',
    '.md', '.markdown', '.tex', '.csv',
    '.html', '.htm', '.xml', '.json', '.yaml', '.yml'
}

SPREADSHEET_EXTENSIONS = {'.xlsx', '.xls', '.ods', '.csv'}

DATABASE_EXTENSIONS = {'.accdb', '.mdb', '.sqlite', '.db'}

COMPRESSED_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.tgz'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.m4v'}


# === Geographical Collections ===
# Each collection represents a different fossil site
# Files will be organized into separate directories based on collection
@dataclass
class CollectionConfig:
    """Configuration for a geographical collection/site."""
    name: str                      # Display name
    code: str                      # Short code for folder naming
    prefixes: tuple                # Specimen ID prefixes associated with this collection
    path_keywords: tuple           # Keywords to detect in path (case-insensitive)
    
COLLECTIONS = {
    'las_hoyas': CollectionConfig(
        name="Las Hoyas",
        code="LH",
        prefixes=('LH', 'MUPA', 'YCLH', 'MCCM', 'MCCM-LH', 'ADL', 'MDCLM'),
        path_keywords=('las hoyas', 'lashoyas', 'hoyas', 'colección lh'),
    ),
    'buenache': CollectionConfig(
        name="Buenache",
        code="BUE",
        prefixes=('K-BUE', 'CER-BUE', 'PB'),  # Note: PB may need disambiguation
        path_keywords=('buenache', 'bue', 'cantera'),
    ),
    'montsec': CollectionConfig(
        name="Montsec",
        code="MON",
        prefixes=(),  # Add confirmed prefixes when known
        path_keywords=('montsec', 'monset'),
    ),
}

# All known specimen prefixes (flat set for quick lookup)
ALL_SPECIMEN_PREFIXES = tuple(
    prefix 
    for coll in COLLECTIONS.values() 
    for prefix in coll.prefixes
)


@dataclass
class Config:
    """Main configuration class."""
    
    # === Source directories ===
    source_directories: List[Path] = field(default_factory=lambda: [
        # Add your MUPA and YCLH paths here
        # Path(r"D:\MUPA"),
        # Path(r"D:\YCLH"),
    ])
    
    # === Output directories ===
    output_base_dir: Path = Path(r"C:\Users\AA.5074193\Desktop\phdAlex_data_management\output")
    output_organized_dir: Path = field(init=False)
    output_text_files_dir: Path = field(init=False)
    output_other_files_dir: Path = field(init=False)
    output_pending_review_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.output_organized_dir = self.output_base_dir / "organized"
        self.output_text_files_dir = self.output_base_dir / "text_files"
        self.output_other_files_dir = self.output_base_dir / "other_files"
        self.output_pending_review_dir = self.output_base_dir / "pending_review"
    
    # === File extensions (reference the module-level sets) ===
    image_extensions: set = field(default_factory=lambda: IMAGE_EXTENSIONS.copy())
    image_with_metadata: set = field(default_factory=lambda: IMAGE_WITH_METADATA_SUPPORT.copy())
    text_extensions: set = field(default_factory=lambda: TEXT_EXTENSIONS.copy())
    spreadsheet_extensions: set = field(default_factory=lambda: SPREADSHEET_EXTENSIONS.copy())
    database_extensions: set = field(default_factory=lambda: DATABASE_EXTENSIONS.copy())
    compressed_extensions: set = field(default_factory=lambda: COMPRESSED_EXTENSIONS.copy())
    video_extensions: set = field(default_factory=lambda: VIDEO_EXTENSIONS.copy())
    
    # === LLM Configuration ===
    # Provider: "gemini" or "github"
    llm_provider: str = field(default_factory=lambda: os.environ.get('LLM_PROVIDER', 'github'))
    llm_requests_per_minute: int = 60  # GitHub Models allows higher RPM
    
    # Gemini settings
    gemini_model: str = field(default_factory=lambda: os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'))
    gemini_api_key_env_var: str = "GEMINI_API_KEY"
    
    # GitHub Models settings (uses OpenAI-compatible API)
    github_model: str = field(default_factory=lambda: os.environ.get('GITHUB_MODEL', 'gpt-4o-mini'))
    github_token_env_var: str = "GITHUB_TOKEN"
    github_models_endpoint: str = "https://models.inference.ai.azure.com"
    
    # Legacy aliases (for backward compatibility)
    @property
    def llm_model(self) -> str:
        if self.llm_provider == 'github':
            return self.github_model
        return self.gemini_model
    
    @property
    def llm_api_key_env_var(self) -> str:
        if self.llm_provider == 'github':
            return self.github_token_env_var
        return self.gemini_api_key_env_var
    
    # === Specimen ID patterns ===
    # Pattern: cccc-dddddddd-pp (characters, separator, digits, plate indicator)
    specimen_id_patterns: List[str] = field(default_factory=lambda: [
        r'^([A-Za-z]{2,6})[-_\s]?(\d{6,10})[-_\s]?([ab])?$',  # MUPA-12345678-a
        r'^(\d{6,10})[-_\s]?([ab])?$',  # Just numbers with optional plate
    ])
    
    # === Image standardization ===
    output_image_format: str = '.jpg'
    output_image_quality: int = 95
    
    # === Logging ===
    log_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.output_organized_dir = self.output_base_dir / "organized"
        self.output_text_files_dir = self.output_base_dir / "text_files"
        self.output_other_files_dir = self.output_base_dir / "other_files"
        self.output_pending_review_dir = self.output_base_dir / "pending_review"
        self.log_dir = self.output_base_dir / "logs"
    
    def get_collection_output_dir(self, collection_code: str) -> Path:
        """Get output directory for a specific collection."""
        return self.output_organized_dir / collection_code
    
    # === Excel/Registry files ===
    @property
    def main_registry_path(self) -> Path:
        return self.output_base_dir / "registros" / "anotaciones.xlsx"
    
    @property
    def text_files_registry_path(self) -> Path:
        return self.output_base_dir / "registros" / "archivos_texto.xlsx"
    
    @property
    def other_files_registry_path(self) -> Path:
        return self.output_base_dir / "registros" / "archivos_otros.xlsx"
    
    @property
    def hash_registry_path(self) -> Path:
        return self.output_base_dir / "registros" / "hashes.xlsx"
    
    def ensure_directories_exist(self):
        """Create all output directories if they don't exist."""
        dirs = [
            self.output_base_dir,
            self.output_organized_dir,
            self.output_text_files_dir,
            self.output_other_files_dir,
            self.output_pending_review_dir,
            self.log_dir,
            self.output_base_dir / "registros",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def detect_collection(path: Path = None, prefix: str = None) -> CollectionConfig | None:
    """
    Detect which collection a file belongs to based on path and/or specimen prefix.
    
    Args:
        path: File path to analyze for collection keywords
        prefix: Specimen ID prefix (e.g., 'LH', 'K-BUE', 'ADL')
        
    Returns:
        CollectionConfig if detected, None otherwise
    """
    # First try prefix matching (most reliable)
    if prefix:
        prefix_upper = prefix.upper()
        for coll in COLLECTIONS.values():
            if prefix_upper in [p.upper() for p in coll.prefixes]:
                return coll
    
    # Then try path keyword matching
    if path:
        path_str = str(path).lower()
        for coll in COLLECTIONS.values():
            for keyword in coll.path_keywords:
                if keyword.lower() in path_str:
                    return coll
    
    return None


# Global config instance
config = Config()
