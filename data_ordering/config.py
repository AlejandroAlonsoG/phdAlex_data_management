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
            'Angiosperma', 'Bennettitales', 'Bryophyta', 'Charophyceae', 'Equisetopsida',
            'Eudicotyledoneae', 'Marchantiopsida', 'Pinales', 'Pinopsida',
            'Planta', 'Pteridophyta', 'Pterophyta'
        }),
    },
    'Insects': {
        'name': 'Insects',
        'name_es': 'Insectos',
        'folder': 'Insects',
        'classes': frozenset({
            'Coleoptera', 'Insecta'
        }),
    },
    'Arthropoda': {
        'name': 'Arthropoda (Non-Insecta)',
        'name_es': 'Artrópodos (sin Insectos)',
        'folder': 'Arthropoda',
        'classes': frozenset({
            'Arachnida', 'Branchiopoda', 'Crustacea', 'Diplopoda', 'Heteropoda',
            'Malacostraca', 'Ostracoda'
        }),
    },
    'Mollusca': {
        'name': 'Mollusca',
        'name_es': 'Moluscos',
        'folder': 'Mollusca',
        'classes': frozenset({
            'Bivalvia', 'Gastropoda', 'Mollusca'
        }),
    },
    'Vermes': {
        'name': 'Vermes',
        'name_es': 'Gusanos',
        'folder': 'Vermes',
        'classes': frozenset({
            'Clitellata', 'Nematoda'
        }),
    },
    'Pisces': {
        'name': 'Pisces',
        'name_es': 'Peces',
        'folder': 'Pisces',
        'classes': frozenset({
            'Actinopterygii', 'Chondrichthyes', 'Lepisosteiforme', 'Osteichthyes', 'Pycnodontiform', 'Sarcopterygii'
        }),
    },
    'Tetrapoda': {
        'name': 'Tetrapoda',
        'name_es': 'Tetrápodos',
        'folder': 'Tetrapoda',
        'classes': frozenset({
            'Amphibia', 'Aves', 'Mammalia', 'Reptilia', 'Sauropsida',
            'Tetrapoda', 'Testudines', 'Vertebrata'
        }),
    },
    'Ichnofossils': {
        'name': 'Ichnofossils',
        'name_es': 'Icnofósiles',
        'folder': 'Ichnofossils',
        'classes': frozenset({
            'Coprolitos', 'icnofósil', 'icnofósiles'
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
# For formats NOT in this set, ImageMagick will be used as a fallback.
# RAW formats (.nef, .cr2, .orf, etc.) are converted to JPEG by ImageMagick.
PIL_CONVERTIBLE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif',
    '.webp', '.psd',  # Pillow reads flattened PSD
    '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raw',  # RAW formats (via ImageMagick)
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
        prefixes=('LH', 'MCLM', 'MCLM-LH', 'MCCM', 'MCCMLH', 'MCCM-LH', 'ADR'),
        path_keywords=('las hoyas', 'lashoyas', 'hoyas', 'colección lh'),
    ),
    'buenache': CollectionConfig(
        name="Buenache",
        code="BUE",
        prefixes=('K-BUE', 'CER-BUE'),
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
    source_directories: List[Path] = field(default_factory=lambda: [])
    
    # === Output directories ===
    output_base_dir: Optional[Path] = None  # Set by CLI/orchestrator
    output_organized_dir: Path = field(init=False)
    output_text_files_dir: Path = field(init=False)
    output_other_files_dir: Path = field(init=False)
    output_pending_review_dir: Path = field(init=False)
    
    # === File extensions (reference the module-level sets) ===
    image_extensions: set = field(default_factory=lambda: IMAGE_EXTENSIONS.copy())
    image_with_metadata: set = field(default_factory=lambda: IMAGE_WITH_METADATA_SUPPORT.copy())
    text_extensions: set = field(default_factory=lambda: TEXT_EXTENSIONS.copy())
    spreadsheet_extensions: set = field(default_factory=lambda: SPREADSHEET_EXTENSIONS.copy())
    database_extensions: set = field(default_factory=lambda: DATABASE_EXTENSIONS.copy())
    compressed_extensions: set = field(default_factory=lambda: COMPRESSED_EXTENSIONS.copy())
    video_extensions: set = field(default_factory=lambda: VIDEO_EXTENSIONS.copy())
    
    # === LLM Configuration ===
    # Provider: "gemini", "github", or "local"
    llm_provider: str = field(default_factory=lambda: os.environ.get('LLM_PROVIDER', 'github'))
    llm_requests_per_minute: int = 60  # GitHub Models allows higher RPM TODO sacar a un parámetro en condiciones
    
    # Gemini settings
    gemini_model: str = field(default_factory=lambda: os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'))
    gemini_api_key_env_var: str = "GEMINI_API_KEY"
    
    # GitHub Models settings (uses OpenAI-compatible API)
    github_model: str = field(default_factory=lambda: os.environ.get('GITHUB_MODEL', 'gpt-4o-mini'))
    github_token_env_var: str = "GITHUB_TOKEN"
    github_models_endpoint: str = "https://models.inference.ai.azure.com"
    
    # Local LLM settings (LM Studio or any OpenAI-compatible local server)
    local_llm_base_url: str = field(default_factory=lambda: os.environ.get('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:1234/v1'))
    local_llm_model: str = field(default_factory=lambda: os.environ.get('LOCAL_LLM_MODEL', 'local-model'))
    local_llm_api_key: str = field(default_factory=lambda: os.environ.get('LOCAL_LLM_API_KEY', 'lm-studio'))
    local_llm_timeout: float = 120.0  # Local models can be slower
    
    # Legacy aliases (for backward compatibility)
    @property
    def llm_model(self) -> str:
        if self.llm_provider == 'local':
            return self.local_llm_model
        if self.llm_provider == 'github':
            return self.github_model
        return self.gemini_model
    
    @property
    def llm_api_key_env_var(self) -> str:
        if self.llm_provider == 'local':
            return 'LOCAL_LLM_API_KEY'
        if self.llm_provider == 'github':
            return self.github_token_env_var
        return self.gemini_api_key_env_var
    
    # === Logging ===
    log_dir: Optional[Path] = field(init=False, default=None)
    
    def __post_init__(self):
        # Only initialize derived paths if output_base_dir is set
        if self.output_base_dir:
            self.output_base_dir = Path(self.output_base_dir)  # Ensure it's a Path object
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
        if not self.output_base_dir:
            raise ValueError("output_base_dir must be set before creating directories")
        
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
