"""
LLM Integration Module v3 for the data ordering tool.

MULTI-STAGE APPROACH:
1. PATH ANALYSIS: Analyze directory path ONLY (no filenames) to extract:
   - taxonomic_class, genus, campaign_year, specimen_id (if visible in path)
   
2. FILENAME REGEX GENERATION: Given sample filenames, generate regex patterns for:
   - specimen_id, campaign_year, taxonomic info (if encoded in filenames)
   
3. REGEX APPLICATION & REFINEMENT:
   - Apply regex to all files
   - Refine if partial failures
   
4. PER-FILE FALLBACK: For files where regex completely fails:
   - Call LLM per-file to extract fields directly

5. COMPARISON: Merge path info with filename-extracted info
"""
import os
import time
import logging
import json
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .config import config


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PathAnalysis:
    """
    Result of LLM analysis for a DIRECTORY PATH only.
    No filenames used - purely from the path structure.
    """
    # Macroclass (top-level category)
    macroclass: Optional[str] = None  # e.g., "Botany", "Arthropoda", "Pisces"
    
    # Taxonomic classification (scientific name)
    taxonomic_class: Optional[str] = None  # e.g., "Mollusca", "Insecta", "Crustacea"
    
    # Genus or more specific determination
    genus: Optional[str] = None  # e.g., "Bivalvia", "Coleoptera", "Delclosia"
    
    # Campaign year (if present in path)
    campaign_year: Optional[int] = None  # e.g., 2018, 2019
    
    # Specimen ID (if directly visible in path, not regex)
    specimen_id: Optional[str] = None  # e.g., "LH-12345" if path contains it
    
    # Collection/site code
    collection_code: Optional[str] = None  # "LH", "BUE", "MON"
    
    # The directory path that was analyzed
    directory_path: str = ""
    
    # Confidence in the analysis (0.0 - 1.0)
    confidence: float = 0.0
    
    # Raw response for debugging
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'macroclass': self.macroclass,
            'taxonomic_class': self.taxonomic_class,
            'genus': self.genus,
            'campaign_year': self.campaign_year,
            'specimen_id': self.specimen_id,
            'collection_code': self.collection_code,
            'directory_path': self.directory_path,
            'confidence': self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PathAnalysis':
        return cls(
            macroclass=data.get('macroclass'),
            taxonomic_class=data.get('taxonomic_class'),
            genus=data.get('genus'),
            campaign_year=data.get('campaign_year'),
            specimen_id=data.get('specimen_id'),
            collection_code=data.get('collection_code'),
            directory_path=data.get('directory_path', ''),
            confidence=data.get('confidence', 0.0),
        )


@dataclass
class FilenameRegexResult:
    """
    Result of LLM analysis to generate regex patterns for filenames.
    Each regex uses named groups to extract specific fields.
    """
    # Master regex pattern with named groups
    # e.g., r"(?P<collection>LH)[-\s]?(?P<specimen_id>\d{3,8})(?P<plate>[ab])?"
    master_regex: Optional[str] = None
    
    # Individual regex patterns (if master doesn't work)
    specimen_id_regex: Optional[str] = None
    campaign_year_regex: Optional[str] = None
    taxonomic_regex: Optional[str] = None
    
    # What fields can be extracted
    extractable_fields: List[str] = field(default_factory=list)
    
    # Sample filenames used
    sample_filenames: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0
    
    # Raw response
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'master_regex': self.master_regex,
            'specimen_id_regex': self.specimen_id_regex,
            'campaign_year_regex': self.campaign_year_regex,
            'taxonomic_regex': self.taxonomic_regex,
            'extractable_fields': self.extractable_fields,
            'sample_filenames': self.sample_filenames,
            'confidence': self.confidence,
        }


@dataclass
class FileAnalysis:
    """
    Result of per-file LLM analysis (fallback when regex fails).
    """
    filename: str = ""
    specimen_id: Optional[str] = None
    campaign_year: Optional[int] = None
    taxonomic_class: Optional[str] = None
    genus: Optional[str] = None
    collection_code: Optional[str] = None
    confidence: float = 0.0
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'specimen_id': self.specimen_id,
            'campaign_year': self.campaign_year,
            'taxonomic_class': self.taxonomic_class,
            'genus': self.genus,
            'collection_code': self.collection_code,
            'confidence': self.confidence,
        }


@dataclass  
class DirectoryAnalysis:
    """
    Combined result of all analysis stages for a directory.
    Backwards compatible with v2.
    """
    # From path analysis
    taxonomic_class: Optional[str] = None
    determination: Optional[str] = None  # genus in new terminology
    collection_code: Optional[str] = None
    campaign_year: Optional[int] = None
    
    # Regex for extracting specimen ID from filenames
    specimen_id_regex: Optional[str] = None
    
    # Confidence in the analysis
    confidence: float = 0.0
    
    # Directory info
    directory_path: Optional[str] = None
    sample_filenames: List[str] = field(default_factory=list)
    
    # Raw response for debugging
    raw_response: Optional[str] = None
    
    # New: detailed results from each stage
    path_analysis: Optional[PathAnalysis] = None
    regex_result: Optional[FilenameRegexResult] = None
    file_analyses: Dict[str, FileAnalysis] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'taxonomic_class': self.taxonomic_class,
            'determination': self.determination,
            'collection_code': self.collection_code,
            'campaign_year': self.campaign_year,
            'specimen_id_regex': self.specimen_id_regex,
            'confidence': self.confidence,
            'directory_path': self.directory_path,
            'sample_filenames': self.sample_filenames,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DirectoryAnalysis':
        return cls(
            taxonomic_class=data.get('taxonomic_class'),
            determination=data.get('determination'),
            collection_code=data.get('collection_code'),
            campaign_year=data.get('campaign_year'),
            specimen_id_regex=data.get('specimen_id_regex'),
            confidence=data.get('confidence', 0.0),
            directory_path=data.get('directory_path'),
            sample_filenames=data.get('sample_filenames', []),
        )
    
    def extract_specimen_id(self, filename: str) -> Optional[str]:
        """Apply regex to extract specimen ID from a filename."""
        if not self.specimen_id_regex:
            return None
        
        try:
            pattern = re.compile(self.specimen_id_regex, re.IGNORECASE)
            match = pattern.search(filename)
            if match:
                # Try named group first
                try:
                    return match.group('specimen_id')
                except IndexError:
                    pass
                # Then first capture group
                return match.group(1) if match.groups() else match.group(0)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{self.specimen_id_regex}': {e}")
        
        return None


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 15):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self.request_count = 0
        self.window_start = time.time()
    
    def wait_if_needed(self):
        """Wait if we're exceeding the rate limit."""
        current_time = time.time()
        
        if current_time - self.window_start >= 60:
            self.request_count = 0
            self.window_start = current_time
        
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.window_start = time.time()
        
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1


# =============================================================================
# PROMPTS - Spanish-aware taxonomic understanding
# =============================================================================

PATH_ANALYSIS_PROMPT = """You are an expert paleontologist analyzing directory paths from a fossil image database.

Your task: Extract taxonomic and collection information ONLY from the directory path structure.
DO NOT analyze filenames - focus purely on the directory path.

=== CLASSIFICATION SYSTEM ===
We use a 3-level hierarchy. These three levels MUST BE DIFFERENT from each other:

1. MACROCLASS (fixed categories - pick exactly one):
   - Botany: All plant life and algae (Angiosperma, Bryophyta, Pteridophyta, Pinales, etc.)
   - Insects: Insects and arthropods (Insecta, Coleoptera)
   - Arthropoda: Arthropods WITHOUT insects (Arachnida, Crustacea, Diplopoda, Malacostraca, Ostracoda, Branchiopoda, Heteropoda)
   - Mollusca: Mollusks (Bivalvia, Gastropoda, Mollusca)
   - Vermes: Worms (Clitellata, Nematoda)
   - Pisces: Fish (vertebrates with fins and gills)
   - Tetrapoda: Land vertebrates with limbs (amphibians, reptiles, birds, mammals)
   - Ichnofossils: Trace fossils (footprints, burrows, coprolites - not body fossils)

2. CLASS: The biological taxonomic class (e.g., Insecta, Malacostraca, Bivalvia, Actinopterygii, Reptilia, etc.)
   - This should be the actual taxonomic class rank from biological classification
   - Translate Spanish terms to scientific Latin names

3. DETERMINATION: Any taxonomic level BELOW class (order, family, genus, species, etc.)
   - The most specific taxonomic information available in the path
   - If the path only goes to class level, leave this null

IMPORTANT RULES:
- Macroclass, class, and determination CANNOT be the same value
- The path may contain Spanish terms - translate them to scientific names
- If you can only determine the macroclass, leave class and determination as null
- If you can determine macroclass and class but nothing more specific, leave determination as null

=== COLLECTION SITES & SPECIMEN PREFIXES ===
Las Hoyas (LH): Prefixes are LH, MCLM, MCLM-LH, or ADR
Buenache (BUE): Prefixes are K-BUE, CER-BUE
Montsec (MON): More prefixes to be confirmed

=== CAMPAIGN YEARS ===
Look for 4-digit years (2018, 2019, 2020, etc.) in path. Pay special attention to date formats on the path, or folder names indicating the year.

The output schema is enforced automatically. Fill each field based on the rules above."""


def _build_prefix_list_text() -> str:
    prefixes = getattr(config, 'ALL_SPECIMEN_PREFIXES', ()) or ()
    if prefixes:
        return ', '.join(prefixes)
    return 'LH, MCLM, MCLM-LH, ADR, K-BUE, CER-BUE'


def get_filename_regex_prompt() -> str:
    """Return the filename-regex prompt with prefixes injected from config."""
    prefix_list = _build_prefix_list_text()
    return f"""You are an expert at creating regex patterns for parsing fossil specimen filenames.

Given sample filenames, create regex patterns with NAMED GROUPS to extract:
1. specimen_id: The unique specimen identifier in format: PREFIX-NUMERIC[-PLATE]
   Examples: "LH-12345", "MCLM-87654321", "ADR-999-a", "K-BUE-123-b"
2. campaign_year: Year of collection if present (e.g., 2019)
3. plate: Plate indicator if present (a, A, b, B, aB, Ab, AB)

SPECIMEN ID STRUCTURE:
- Prefix: {prefix_list}. Only these prefixes are considered specimen IDs; other prefixes are likely camera-generated and should not be considered specimen IDs.
- Numeric: 1-10 digits (e.g., 5, 12345, 87654321)
- Plate: Optional - only a, A, b, B, aB, Ab, AB (lowercase preferred)
- Separators: space, hyphen, underscore, forward slash, or none

COMMON FILENAME PATTERNS:
- "LH-12345.jpg" → specimen_id = "LH-12345"
- "LH 12345a.tif" → specimen_id = "LH-12345-a"
- "MCLM-87654321-b.jpg" → specimen_id = "MCLM-87654321-b"
- "K-BUE_999_AB.tif" → specimen_id = "K-BUE-999-ab"
- "specimen_2019_LH_12345.jpg" → campaign_year = 2019, specimen_id = "LH-12345"
- "DSC00001.jpg" → likely camera-generated, not a specimen ID

REGEX GUIDELINES:
- Use Python regex syntax
- Use named groups: (?P<name>pattern)
- Make patterns flexible for spaces/hyphens/underscores: [-\\s_]?
- Handle optional plate indicators: (?P<plate>[aAbB]{1,2})?
- Example: r"(?P<specimen_id>(LH|MCLM|ADR)[-\\s_]?\\d{1,10})(?:\\s*(?P<plate>[aAbB]{1,2}))?"

The output schema is enforced automatically. Fill each field based on the guidelines above."""


def get_file_analysis_prompt() -> str:
    """Return the file-analysis prompt with prefixes from config injected."""
    prefixes = _build_prefix_list_text()
    return f"""You are an expert paleontologist. Extract information from this single fossil image filename.

This is a FALLBACK analysis when regex patterns failed. Be as accurate as possible.

SPECIMEN ID FORMATS:
Specimen IDs follow the structure: PREFIX-NUMERIC[-PLATE]
- Prefix options: {prefixes}
- Numeric: 1-10 digits (e.g., 5, 12345, 87654321)
- Plate: Optional single letter or pair (a, A, b, B, aB, Ab, AB)
- Examples: "LH-12345", "MCLM-87654321-a", "K-BUE-999-b"

Examples of camera numbers (NOT specimen IDs):
- DSC00001, IMG_1234, DSCN5678, _MG_7890, DJI_0042

The output schema is enforced automatically. Fill each field based on the formats above."""


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================
# These schemas enforce the exact JSON shape for each stage.
# Stored in Gemini-native format ("nullable": true for optional fields).
# GitHubModelsClient transforms them to OpenAI format at call time:
#   - "nullable": true  →  "anyOf": [{"type": T}, {"type": "null"}]
#   - Injects "additionalProperties": false (required by OpenAI strict mode)

PATH_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "macroclass": {
            "type": "string",
            "nullable": True,
            "enum": ["Botany", "Insects", "Arthropoda", "Mollusca", "Vermes", "Pisces", "Tetrapoda", "Ichnofossils"],
            "description": "Top-level category. Botany=plants/algae, Insects=insects/Coleoptera, Arthropoda=arthropods without insects, Mollusca=mollusks, Vermes=worms, Pisces=fish, Tetrapoda=amphibians/reptiles/birds/mammals, Ichnofossils=trace fossils. Null if undetermined."
        },
        "taxonomic_class": {
            "type": "string",
            "nullable": True,
            "description": "Biological taxonomic class in scientific Latin (e.g. Insecta, Malacostraca, Bivalvia, Actinopterygii, Reptilia). Translate Spanish terms to Latin. Must differ from macroclass. Null if only macroclass is determinable."
        },
        "genus": {
            "type": "string",
            "nullable": True,
            "description": "Most specific taxonomic determination below class level (order, family, genus, or species). Must differ from both macroclass and taxonomic_class. Null if nothing below class is determinable."
        },
        "campaign_year": {
            "type": "integer",
            "nullable": True,
            "description": "4-digit campaign/excavation year (e.g. 2018, 2019) if present in the directory path. Null otherwise."
        },
        "specimen_id": {
            "type": "string",
            "nullable": True,
            "description": "Specimen identifier if directly visible in the directory path in format PREFIX-NUMERIC[-PLATE] (e.g. LH-12345, MCLM-87654321-a). Null if not present in path."
        },
        "collection_code": {
            "type": "string",
            "enum": ["LH", "BUE", "MON"],
            "description": "Collection site: LH=Las Hoyas, BUE=Buenache, MON=Montsec. If not determinable, asume LH."
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the overall analysis from 0.0 to 1.0"
        }
    },
    "required": ["macroclass", "taxonomic_class", "genus", "campaign_year", "specimen_id", "collection_code", "confidence"],
}

FILENAME_REGEX_SCHEMA = {
    "type": "object",
    "properties": {
        "master_regex": {
            "type": "string",
            "nullable": True,
            "description": "Single Python regex with named groups (?P<name>...) to extract all fields at once. Use [-\\s]? for flexible separators. Null if a single pattern is not feasible."
        },
        "specimen_id_regex": {
            "type": "string",
            "nullable": True,
            "description": "Python regex with a capture group to extract specimen ID only. E.g. r'(LH[-\\s]?\\d{3,8})' or r'(K-Bue[-\\s]?\\d+)'. Null if specimen ID is not extractable."
        },
        "campaign_year_regex": {
            "type": "string",
            "nullable": True,
            "description": "Python regex to extract a 4-digit campaign year from filenames. Null if year is not encoded in filenames."
        },
        "extractable_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of field names that can be extracted from these filenames (e.g. 'specimen_id', 'plate', 'campaign_year')"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the regex patterns from 0.0 to 1.0"
        },
        "notes": {
            "type": "string",
            "nullable": True,
            "description": "Any important notes about the filename patterns or edge cases"
        }
    },
    "required": ["master_regex", "specimen_id_regex", "campaign_year_regex", "extractable_fields", "confidence", "notes"],
}

FILE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "specimen_id": {
            "type": "string",
            "nullable": True,
            "description": "Extracted specimen ID in format PREFIX-NUMERIC[-PLATE]. Examples: 'LH-12345', 'MCLM-87654321-a', 'K-BUE-999-b', 'ADR-5'. Valid prefixes: LH, MCLM, MCLM-LH, ADR (Las Hoyas); K-BUE, CER-BUE (Buenache). Null if not identifiable."
        },
        "campaign_year": {
            "type": "integer",
            "nullable": True,
            "description": "4-digit campaign year if encoded in the filename. Null otherwise."
        },
        "taxonomic_class": {
            "type": "string",
            "nullable": True,
            "description": "Taxonomic class in scientific Latin if encoded in the filename. Null otherwise."
        },
        "genus": {
            "type": "string",
            "nullable": True,
            "description": "Genus or more specific determination if encoded in the filename. Null otherwise."
        },
        "collection_code": {
            "type": "string",
            "enum": ["LH", "BUE", "MON"],
            "description": "Collection site: LH=Las Hoyas, BUE=Buenache, MON=Montsec. If not determinable, assume LH."
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the analysis from 0.0 to 1.0"
        },
        "is_camera_generated": {
            "type": "boolean",
            "description": "True if filename looks auto-generated by a camera (e.g. DSC00001, IMG_1234, DSCN0001). False otherwise."
        }
    },
    "required": ["specimen_id", "campaign_year", "taxonomic_class", "genus", "collection_code", "confidence", "is_camera_generated"],
}

# Map schema names to schema objects for lookup
_SCHEMAS = {
    "path_analysis": PATH_ANALYSIS_SCHEMA,
    "filename_regex": FILENAME_REGEX_SCHEMA,
    "file_analysis": FILE_ANALYSIS_SCHEMA,
}


def _schema_to_openai(schema: dict) -> dict:
    """Transform a Gemini-native schema to OpenAI strict-mode format.
    
    Conversions:
      - {"type": T, "nullable": true, ...}  →  {"anyOf": [{"type": T, ...}, {"type": "null"}]}
      - Injects "additionalProperties": false on every object
      - Recurses into nested objects and array items
    """
    import copy
    schema = copy.deepcopy(schema)
    return _convert_node(schema)


def _convert_node(node: dict) -> dict:
    """Recursively convert a single schema node."""
    if not isinstance(node, dict):
        return node
    
    node_type = node.get("type")
    
    # Handle nullable fields: pull out "nullable" and wrap in anyOf
    if node.pop("nullable", False):
        # Build the non-null branch (everything except nullable)
        non_null = {k: v for k, v in node.items()}
        non_null = _convert_node(non_null)
        return {"anyOf": [non_null, {"type": "null"}]}
    
    # Handle objects: recurse into properties, add additionalProperties
    if node_type == "object":
        node["additionalProperties"] = False
        if "properties" in node:
            node["properties"] = {
                k: _convert_node(v)
                for k, v in node["properties"].items()
            }
    
    # Handle arrays: recurse into items
    if node_type == "array" and "items" in node:
        node["items"] = _convert_node(node["items"])
    
    return node


def _normalize_extraction(extracted: dict, match: re.Match = None) -> dict:
    """Normalize extracted fields from a regex match to canonical specimen_id format.

    Rules:
    - prefix: leave as extracted
    - numeric part: leave as extracted
    - separators: always use '-'
    - plate: always lowercase
    """
    res = dict(extracted or {})

    # Lowercase plate if present
    if 'plate' in res and res.get('plate'):
        res['plate'] = res['plate'].lower()

    specimen_raw = res.get('specimen_id')
    # Try to parse specimen_id if present
    if specimen_raw:
        m = re.match(r"^\s*(?P<prefix>[A-Za-z0-9\-]+)[-\s_/]*(?P<number>\d{1,10})(?:[-\s_/]*(?P<plate>[A-Za-z]{1,2}))?\s*$",
                     specimen_raw)
        if m:
            prefix = m.group('prefix')
            number = m.group('number')
            plate = m.group('plate').lower() if m.group('plate') else None
            specimen_id = f"{prefix}-{number}" + (f"-{plate}" if plate else "")
            res['specimen_id'] = specimen_id
            res['prefix'] = prefix
            res['number'] = number
            res['plate'] = plate
            return res

    # If specimen_id was not present or parsing failed, try to assemble from components
    prefix = res.get('prefix') or res.get('specimen_prefix') or res.get('collection')
    number = res.get('number') or res.get('numeric') or res.get('id') or res.get('specimen_number')
    if prefix and number:
        plate = res.get('plate') or res.get('p') or None
        if plate:
            plate = plate.lower()
        specimen_id = f"{prefix}-{number}" + (f"-{plate}" if plate else "")
        res['specimen_id'] = specimen_id
        res['prefix'] = prefix
        res['number'] = number
        res['plate'] = plate

    return res


# =============================================================================
# LLM CLIENT BASE
# =============================================================================

class BaseLLMClient:
    """Base class for LLM clients with common functionality."""
    
    def __init__(self, requests_per_minute: int = 15):
        self.rate_limiter = RateLimiter(requests_per_minute)
    
    def _call_api(self, system_prompt: str, user_prompt: str, schema_name: str = None) -> str:
        """Override in subclasses. Returns raw response text.
        
        Args:
            system_prompt: The system/instruction prompt
            user_prompt: The user query
            schema_name: Key into _SCHEMAS dict to enforce structured output.
                         One of: 'path_analysis', 'filename_regex', 'file_analysis'.
        """
        raise NotImplementedError
    
    def _parse_json(self, response_text: str) -> dict:
        """Parse JSON from response.
        
        With structured outputs enabled, the response is guaranteed to be valid
        JSON matching the schema. We still keep a fallback for markdown-wrapped
        responses in case a provider doesn't support structured outputs.
        """
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Fallback: strip markdown code fences if present
            json_text = response_text
            try:
                if '```json' in response_text:
                    json_text = response_text.split('```json')[1].split('```')[0]
                elif '```' in response_text:
                    parts = response_text.split('```')
                    if len(parts) >= 2:
                        json_text = parts[1]
                return json.loads(json_text.strip())
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Response was: {response_text[:500]}")
                return {}
    
    # -------------------------------------------------------------------------
    # STAGE 1: Path Analysis
    # -------------------------------------------------------------------------
    
    def analyze_path(self, directory_path: Path) -> PathAnalysis:
        """
        Analyze ONLY the directory path (no filenames).
        
        Args:
            directory_path: Path to analyze
            
        Returns:
            PathAnalysis with extracted information
        """
        self.rate_limiter.wait_if_needed()
        
        user_prompt = f"""Analyze this directory path from a fossil image database:

Path: {directory_path}

Extract any taxonomic, collection, or campaign information visible in the path structure.
Remember to convert Spanish taxonomic terms to scientific names."""

        try:
            response_text = self._call_api(PATH_ANALYSIS_PROMPT, user_prompt, schema_name='path_analysis')
            data = self._parse_json(response_text)

            # Normalize campaign_year: allow integers or numeric strings (e.g. "2005").
            cy = data.get('campaign_year')
            if cy is not None:
                try:
                    # Some LLMs may return the year as a string or with stray chars
                    cy = int(str(cy).strip())
                except Exception:
                    cy = None

            # If year wasn't extracted, keep the raw response for debugging
            if cy is None:
                logger.debug(f"Path analysis did not include campaign_year. Raw response: {response_text[:1000]}")

            return PathAnalysis(
                macroclass=data.get('macroclass'),
                taxonomic_class=data.get('taxonomic_class'),
                genus=data.get('genus'),
                campaign_year=cy,
                specimen_id=data.get('specimen_id'),
                collection_code=data.get('collection_code', 'LH'),
                directory_path=str(directory_path),
                confidence=float(data.get('confidence', 0)),
                raw_response=response_text,
            )
        except Exception as e:
            logger.error(f"Path analysis error for {directory_path}: {e}")
            return PathAnalysis(
                directory_path=str(directory_path),
                raw_response=str(e),
            )
    
    # -------------------------------------------------------------------------
    # STAGE 2: Filename Regex Generation
    # -------------------------------------------------------------------------
    
    def generate_filename_regex(
        self,
        sample_filenames: List[str],
        directory_path: Path = None,
        refinement_context: str = None
    ) -> FilenameRegexResult:
        """
        Generate regex patterns to extract data from filenames.
        
        Args:
            sample_filenames: List of sample filenames
            directory_path: Optional path for context
            refinement_context: Optional context for refinement
            
        Returns:
            FilenameRegexResult with regex patterns
        """
        self.rate_limiter.wait_if_needed()
        
        user_prompt = f"""Analyze these filenames from a fossil image database:

Sample filenames:
{chr(10).join(f"- {fn}" for fn in sample_filenames[:20])}

{f"Directory: {directory_path}" if directory_path else ""}

Create regex patterns with named groups to extract specimen_id, year, and plate information."""

        if refinement_context:
            user_prompt += f"\n\n{refinement_context}"

        try:
            response_text = self._call_api(get_filename_regex_prompt(), user_prompt, schema_name='filename_regex')
            data = self._parse_json(response_text)
            
            return FilenameRegexResult(
                master_regex=data.get('master_regex'),
                specimen_id_regex=data.get('specimen_id_regex'),
                campaign_year_regex=data.get('campaign_year_regex'),
                taxonomic_regex=data.get('taxonomic_regex'),
                extractable_fields=data.get('extractable_fields', []),
                sample_filenames=sample_filenames[:20],
                confidence=float(data.get('confidence', 0)),
                raw_response=response_text,
            )
        except Exception as e:
            logger.error(f"Regex generation error: {e}")
            return FilenameRegexResult(
                sample_filenames=sample_filenames[:20],
                raw_response=str(e),
            )
    
    # -------------------------------------------------------------------------
    # STAGE 3: Regex Application & Validation
    # -------------------------------------------------------------------------
    
    def apply_and_validate_regex(
        self,
        regex_result: FilenameRegexResult,
        filenames: List[str],
        max_refinement_attempts: int = 2
    ) -> Tuple[Dict[str, dict], List[str], FilenameRegexResult]:
        """
        Apply regex to filenames and validate.
        
        Args:
            regex_result: The regex patterns to apply
            filenames: All filenames to process
            max_refinement_attempts: Max times to refine
            
        Returns:
            Tuple of (successful extractions, failed filenames, refined regex)
        """
        for attempt in range(max_refinement_attempts + 1):
            regex = regex_result.specimen_id_regex or regex_result.master_regex
            if not regex:
                logger.warning("No regex pattern available")
                return {}, filenames, regex_result
            
            successes = {}
            failures = []
            
            try:
                pattern = re.compile(regex, re.IGNORECASE)
                
                for fn in filenames:
                    match = pattern.search(fn)
                    if match:
                        # Extract all named groups or positional groups
                        if match.groupdict():
                            extracted = {k: v for k, v in match.groupdict().items() if v}
                        else:
                            extracted = {
                                'specimen_id': match.group(1) if match.groups() else match.group(0)
                            }
                        # Normalize the extraction to canonical specimen_id format
                        normalized = _normalize_extraction(extracted, match)
                        successes[fn] = normalized
                    else:
                        failures.append(fn)
                        
            except re.error as e:
                logger.warning(f"Invalid regex '{regex}': {e}")
                failures = filenames
            
            # Calculate success rate
            success_rate = len(successes) / len(filenames) if filenames else 0
            logger.info(f"Regex attempt {attempt + 1}: {len(successes)}/{len(filenames)} matched ({success_rate:.1%})")
            
            # If good enough, return
            if success_rate >= 0.7:
                return successes, failures, regex_result
            
            # If all failed or last attempt, stop trying
            if success_rate == 0 or attempt >= max_refinement_attempts:
                break
            
            # Refine regex
            logger.info(f"Refining regex (attempt {attempt + 1})")
            refinement_context = f"""IMPORTANT: Your previous regex pattern had issues.

Previous pattern: {regex}

Files where it FAILED ({len(failures)} files):
{chr(10).join(f"- {fn}" for fn in failures[:10])}

Files where it SUCCEEDED ({len(successes)} files):
{chr(10).join(f"- {fn} → {data}" for fn, data in list(successes.items())[:5])}

Please provide a CORRECTED regex pattern that will match ALL filenames."""

            regex_result = self.generate_filename_regex(
                filenames[:20],
                refinement_context=refinement_context
            )
        
        return successes, failures, regex_result
    
    # -------------------------------------------------------------------------
    # STAGE 4: Per-file Fallback Analysis
    # -------------------------------------------------------------------------
    
    def analyze_file(self, filename: str, directory_context: str = None) -> FileAnalysis:
        """
        Analyze a single filename when regex fails.
        
        Args:
            filename: The filename to analyze
            directory_context: Optional directory path for context
            
        Returns:
            FileAnalysis with extracted information
        """
        self.rate_limiter.wait_if_needed()
        
        user_prompt = f"""Analyze this fossil image filename:

Filename: {filename}
{f"From directory: {directory_context}" if directory_context else ""}

Extract any specimen ID, year, or taxonomic information encoded in the filename."""

        try:
            response_text = self._call_api(get_file_analysis_prompt(), user_prompt, schema_name='file_analysis')
            data = self._parse_json(response_text)
            
            return FileAnalysis(
                filename=filename,
                specimen_id=data.get('specimen_id'),
                campaign_year=data.get('campaign_year'),
                taxonomic_class=data.get('taxonomic_class'),
                genus=data.get('genus'),
                collection_code=data.get('collection_code', 'LH'),
                confidence=float(data.get('confidence', 0)),
                raw_response=response_text,
            )
        except Exception as e:
            logger.error(f"File analysis error for {filename}: {e}")
            return FileAnalysis(
                filename=filename,
                raw_response=str(e),
            )
    
    def analyze_files_batch(
        self,
        filenames: List[str],
        directory_context: str = None,
        progress_callback: callable = None
    ) -> Dict[str, FileAnalysis]:
        """
        Analyze multiple files one by one (fallback mode).
        
        Args:
            filenames: Files to analyze
            directory_context: Optional directory path
            progress_callback: Called with (current, total)
            
        Returns:
            Dict mapping filename to FileAnalysis
        """
        results = {}
        total = len(filenames)
        
        for i, fn in enumerate(filenames):
            results[fn] = self.analyze_file(fn, directory_context)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    # -------------------------------------------------------------------------
    # FULL DIRECTORY ANALYSIS (combines all stages)
    # -------------------------------------------------------------------------
    
    def analyze_directory(
        self,
        directory_path: Path,
        sample_filenames: List[str] = None,
        all_filenames: List[str] = None,
        max_samples: int = 20,
        max_refinement_attempts: int = 2,
        enable_per_file_fallback: bool = True,
        progress_callback: callable = None
    ) -> DirectoryAnalysis:
        """
        Full multi-stage directory analysis.
        
        Args:
            directory_path: Path to analyze
            sample_filenames: Sample filenames for regex generation
            all_filenames: All filenames to validate regex against
            max_samples: Max samples for regex generation
            max_refinement_attempts: Max regex refinement attempts
            enable_per_file_fallback: Whether to use per-file LLM fallback
            progress_callback: Progress callback
            
        Returns:
            DirectoryAnalysis with combined results
        """
        # Get filenames if not provided
        if sample_filenames is None:
            try:
                all_files = list(directory_path.iterdir())
                files = [f.name for f in all_files if f.is_file()]
                sample_filenames = files[:max_samples]
                all_filenames = all_filenames or files
            except Exception as e:
                logger.warning(f"Could not list directory {directory_path}: {e}")
                sample_filenames = []
                all_filenames = []
        
        all_filenames = all_filenames or sample_filenames
        
        # STAGE 1: Path analysis
        logger.info(f"Stage 1: Analyzing path {directory_path}")
        path_analysis = self.analyze_path(directory_path)
        
        # STAGE 2: Generate filename regex
        logger.info(f"Stage 2: Generating regex from {len(sample_filenames)} samples")
        regex_result = None
        if sample_filenames:
            regex_result = self.generate_filename_regex(
                sample_filenames,
                directory_path
            )
        
        # STAGE 3: Apply and validate regex
        successes = {}
        failures = []
        if regex_result and all_filenames:
            logger.info(f"Stage 3: Applying regex to {len(all_filenames)} files")
            successes, failures, regex_result = self.apply_and_validate_regex(
                regex_result,
                all_filenames,
                max_refinement_attempts
            )
        
        # STAGE 4: Per-file fallback for failures
        file_analyses = {}
        if enable_per_file_fallback and failures:
            logger.info(f"Stage 4: Per-file fallback for {len(failures)} failed files")
            file_analyses = self.analyze_files_batch(
                failures,
                str(directory_path),
                progress_callback
            )
        
        # Combine results into DirectoryAnalysis
        result = DirectoryAnalysis(
            taxonomic_class=path_analysis.taxonomic_class,
            determination=path_analysis.genus,
            collection_code=path_analysis.collection_code,
            campaign_year=path_analysis.campaign_year,
            specimen_id_regex=regex_result.specimen_id_regex if regex_result else None,
            confidence=path_analysis.confidence,
            directory_path=str(directory_path),
            sample_filenames=sample_filenames[:10],
            path_analysis=path_analysis,
            regex_result=regex_result,
            file_analyses=file_analyses,
        )
        
        return result


# =============================================================================
# GITHUB MODELS CLIENT
# =============================================================================

class GitHubModelsClient(BaseLLMClient):
    """
    Client for GitHub Models API (OpenAI-compatible).
    """
    
    def __init__(
        self,
        token: str = None,
        model: str = None,
        endpoint: str = None,
        requests_per_minute: int = None
    ):
        super().__init__(requests_per_minute or config.llm_requests_per_minute)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.token = token or os.environ.get(config.github_token_env_var)
        if not self.token:
            raise ValueError(
                f"GitHub token required. Set {config.github_token_env_var} env var or pass token."
            )
        
        self.model_name = model or config.github_model
        self.endpoint = endpoint or config.github_models_endpoint
        
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        logger.info(f"Initialized GitHub Models client with model {self.model_name}")
    
    def _call_api(self, system_prompt: str, user_prompt: str, schema_name: str = None) -> str:
        """Make API call with structured output enforcement.
        
        Uses OpenAI Structured Outputs (json_schema with strict: true) when a
        schema_name is provided, guaranteeing the response matches the schema.
        Falls back to basic json_object mode if no schema is given.
        """
        logger.debug(f"[LLM_REQUEST] schema={schema_name} | prompt={user_prompt[:300]}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if schema_name and schema_name in _SCHEMAS:
            # Transform Gemini-native schema to OpenAI strict format:
            #   - "nullable": true  →  "anyOf": [{"type": T, ...}, {"type": "null"}]
            #   - inject "additionalProperties": false (required by strict mode)
            schema = _schema_to_openai(_SCHEMAS[schema_name])
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        else:
            # Fallback for calls without a defined schema
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        
        result = response.choices[0].message.content
        logger.debug(f"[LLM_RESPONSE] schema={schema_name} | response={result[:500]}")
        return result


# =============================================================================
# GEMINI CLIENT
# =============================================================================

class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini API with structured output and thinking support.
    """
    
    # Thinking levels mapped to task complexity
    THINKING_LEVELS = {
        'path_analysis': 'MINIMAL',           # Quick thinking - fast path analysis
        'filename_regex': 'MINIMAL',          # Quick thinking - fast regex generation
        'file_analysis': 'MINIMAL',           # Quick thinking - simple filename parsing
    }
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        requests_per_minute: int = None,
        enable_thinking: bool = True
    ):
        super().__init__(requests_per_minute or config.llm_requests_per_minute)
        
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-genai not installed. Install with: pip install -U google-genai"
            )
        
        self.api_key = api_key or os.environ.get(config.llm_api_key_env_var)
        if not self.api_key:
            raise ValueError(f"API key required. Set {config.llm_api_key_env_var} env var")
        
        self.model_name = model or config.llm_model
        self.enable_thinking = enable_thinking
        
        # Initialize the new google.genai client with API key
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(
            f"Initialized Gemini client with model {self.model_name} "
            f"(thinking: {'enabled' if enable_thinking else 'disabled'})"
        )
    
    def _call_api(self, system_prompt: str, user_prompt: str, schema_name: str = None) -> str:
        """Make API call with structured output and thinking support.
        
        Uses Gemini's response_mime_type and response_json_schema parameters when a schema_name is provided,
        guaranteeing the response matches the schema. Applies appropriate thinking
        level based on task complexity (schema_name) to limit model reasoning.
        """
        logger.debug(f"[LLM_REQUEST] schema={schema_name} | prompt={user_prompt[:300]}")
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Build the config dictionary
        config_kwargs = {
            "temperature": 0.1,  # Low temperature for deterministic outputs
            "max_output_tokens": 8192,
        }
        
        # Add structured output configuration if schema available
        if schema_name and schema_name in _SCHEMAS:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = _SCHEMAS[schema_name]
            logger.debug(f"[STRUCTURED_OUTPUT] schema={schema_name}")
        
        # Add thinking configuration if enabled (limits how much the model thinks)
        if self.enable_thinking and schema_name in self.THINKING_LEVELS:
            thinking_level = self.THINKING_LEVELS[schema_name]
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
            )
            logger.debug(f"[THINKING] schema={schema_name} | level={thinking_level}")
        
        # Create the GenerateContentConfig with all parameters
        generate_config = types.GenerateContentConfig(**config_kwargs)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=generate_config
        )
        
        result = response.text
        logger.debug(f"[LLM_RESPONSE] schema={schema_name} | response={result[:500]}")
        return result


# =============================================================================
# LOCAL LLM CLIENT (LM Studio / llama-cpp-python / any OpenAI-compatible)
# =============================================================================

class LocalLLMClient(BaseLLMClient):
    """
    Client for a locally-hosted LLM via an OpenAI-compatible API.

    Works with LM Studio, llama-cpp-python, ollama, or any server that
    exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Default settings target LM Studio at ``http://127.0.0.1:1234/v1``.
    Override via constructor args, ``config``, or environment variables.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        api_key: str = None,
        requests_per_minute: int = None,
        timeout: float = None,
    ):
        super().__init__(requests_per_minute or 120)  # local = no rate-limit concern

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.base_url = (
            base_url
            or config.local_llm_base_url
        )
        self.model_name = (
            model
            or config.local_llm_model
        )
        self.timeout = timeout or config.local_llm_timeout
        _api_key = api_key or config.local_llm_api_key

        # Ensure localhost traffic bypasses any HTTP proxy (e.g. university networks)
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        local_host = parsed.hostname or "127.0.0.1"
        for var in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(var, "")
            if local_host not in current:
                os.environ[var] = f"{current},{local_host}" if current else local_host

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=_api_key,
            timeout=self.timeout,
        )

        logger.info(
            f"Initialized LocalLLMClient: {self.base_url} "
            f"(model: {self.model_name})"
        )

    # --------------------------------------------------------------------- API
    def _call_api(self, system_prompt: str, user_prompt: str, schema_name: str = None) -> str:
        """Call the local OpenAI-compatible endpoint.

        If *schema_name* is provided **and** the server supports
        ``response_format`` with ``json_schema``, structured output is
        requested.  Otherwise we fall back to ``json_object`` mode and
        finally to free-text if even that is unsupported.
        """
        logger.debug(f"[LLM_REQUEST] schema={schema_name} | prompt={user_prompt[:300]}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Try structured output first (LM Studio ≥0.3 supports it)
        if schema_name and schema_name in _SCHEMAS:
            schema = _schema_to_openai(_SCHEMAS[schema_name])
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                result = response.choices[0].message.content
                logger.debug(f"[LLM_RESPONSE] schema={schema_name} | response={result[:500]}")
                return result
            except Exception as e:
                # Server may not support json_schema; fall back gracefully
                logger.debug(f"Structured output not supported, falling back: {e}")

        # Fallback: request plain JSON
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Last resort: no response_format constraint at all
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )

        result = response.choices[0].message.content
        if not result:
            raise ValueError("Empty response from local LLM")
        logger.debug(f"[LLM_RESPONSE] schema={schema_name} | response={result[:500]}")
        return result

    # ----------------------------------------------------------- health check
    def is_server_running(self) -> bool:
        """Check if the local LLM server is reachable."""
        try:
            base = self.base_url.replace("/v1", "")
            resp = requests.get(f"{base}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        # Fallback: try /v1/models endpoint
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_server_info(self) -> dict:
        """Return information about the running server and loaded model."""
        try:
            models = self.client.models.list()
            model_list = [m.id for m in models.data]
            return {
                "status": "running",
                "base_url": self.base_url,
                "models": model_list,
                "configured_model": self.model_name,
            }
        except Exception as e:
            return {
                "status": "error",
                "base_url": self.base_url,
                "error": str(e),
            }

    def wait_for_server(self, timeout: float = 60.0, poll_interval: float = 2.0):
        """Block until the server is reachable (or *timeout* expires)."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_server_running():
                logger.info("Local LLM server is ready")
                return True
            logger.info(f"Waiting for server at {self.base_url}...")
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Server at {self.base_url} did not become available "
            f"within {timeout}s. Make sure LM Studio is running."
        )


# =============================================================================
# MOCK CLIENT (for testing)
# =============================================================================

class MockLLMClient(BaseLLMClient):
    """Mock client for testing without actual API calls.
    
    This mock simulates LLM behavior for testing. It uses simple keyword detection
    to return reasonable responses following the macroclass → class → determination hierarchy.
    """
    
    def __init__(self):
        super().__init__(requests_per_minute=1000)  # No real limiting
    
    def _call_api(self, system_prompt: str, user_prompt: str, schema_name: str = None) -> str:
        """Return mock responses based on content."""
        user_lower = user_prompt.lower()
        
        # Detect collection from path
        collection = "LH"  # Default to LH if not detected
        if any(x in user_lower for x in ['lh', 'las hoyas', 'colección lh', 'mupa', 'yclh']):
            collection = "LH"
        elif any(x in user_lower for x in ['buenache', 'bue', 'k-bue']):
            collection = "BUE"
        elif any(x in user_lower for x in ['montsec', 'mon', 'catalonia', 'lleida']):
            collection = "MON"
        
        # Path analysis mock - simulates smart LLM behavior
        if "path" in user_lower:
            macroclass = None
            tax_class = None
            determination = None
            
            # Detect macroclass and class from keywords
            # Botany
            if any(x in user_lower for x in ['plant', 'flora', 'carofita', 'charophyt', 'helecho', 'angiosperma', 'gimnosperma', 'semilla', 'hoja', 'madera']):
                macroclass = "Botany"
                if 'charophyt' in user_lower or 'carofita' in user_lower:
                    tax_class = "Charophyceae"
                elif 'angiosperma' in user_lower:
                    tax_class = "Magnoliopsida"
                elif 'gimnosperma' in user_lower:
                    tax_class = "Pinopsida"
                elif 'helecho' in user_lower:
                    tax_class = "Polypodiopsida"
                # No default class for plants - let it be null if not specific
            
            # Arthropoda
            elif any(x in user_lower for x in ['artropod', 'arthropod', 'insect', 'crustace', 'arachn', 'araña', 'decapod', 'coleoptera', 'diptera', 'odonata', 'hemiptera', 'delclosia']):
                macroclass = "Arthropoda"
                if any(x in user_lower for x in ['insect', 'coleoptera', 'diptera', 'odonata', 'hemiptera']):
                    tax_class = "Insecta"
                    if 'coleoptera' in user_lower:
                        determination = "Coleoptera"
                    elif 'diptera' in user_lower:
                        determination = "Diptera"
                    elif 'odonata' in user_lower:
                        determination = "Odonata"
                    elif 'hemiptera' in user_lower:
                        determination = "Hemiptera"
                elif any(x in user_lower for x in ['crustace', 'decapod', 'delclosia']):
                    tax_class = "Malacostraca"
                    if 'delclosia' in user_lower:
                        determination = "Delclosia"
                    elif 'decapod' in user_lower:
                        determination = "Decapoda"
                elif any(x in user_lower for x in ['arachn', 'araña']):
                    tax_class = "Arachnida"
            
            # Mollusca y Vermes
            elif any(x in user_lower for x in ['molusco', 'mollusc', 'bivalvo', 'gastropod', 'caracol', 'gusano', 'anelid', 'verme']):
                macroclass = "Mollusca_y_Vermes"
                if any(x in user_lower for x in ['bivalvo', 'almeja']):
                    tax_class = "Bivalvia"
                elif any(x in user_lower for x in ['gastropod', 'caracol']):
                    tax_class = "Gastropoda"
                elif any(x in user_lower for x in ['gusano', 'anelid', 'verme']):
                    tax_class = "Clitellata"
                # If just "molusco", leave class null - not enough info
            
            # Pisces
            elif any(x in user_lower for x in ['peces', 'pez', 'fish', 'osteichthy', 'actinopterygii', 'chondrichthy', 'tiburon', 'raya']):
                macroclass = "Pisces"
                if any(x in user_lower for x in ['actinopterygii', 'osteichthy']):
                    tax_class = "Actinopterygii"
                elif any(x in user_lower for x in ['chondrichthy', 'tiburon', 'raya']):
                    tax_class = "Chondrichthyes"
            
            # Tetrapoda
            elif any(x in user_lower for x in ['anfibio', 'amphibia', 'reptil', 'sauropsida', 'dinosaur', 'ave', 'bird', 'mamif', 'tetrapod']):
                macroclass = "Tetrapoda"
                if any(x in user_lower for x in ['anfibio', 'amphibia']):
                    tax_class = "Amphibia"
                elif any(x in user_lower for x in ['reptil', 'sauropsida', 'dinosaur']):
                    tax_class = "Reptilia"
                    if 'dinosaur' in user_lower:
                        determination = "Dinosauria"
                elif any(x in user_lower for x in ['ave', 'bird', 'pajaro']):
                    tax_class = "Aves"
                elif 'mamif' in user_lower:
                    tax_class = "Mammalia"
            
            # Ichnofossils
            elif any(x in user_lower for x in ['icno', 'ichno', 'huella', 'coprolito', 'madriguera', 'trace']):
                macroclass = "Ichnofossils"
                # Ichnofossils don't have traditional taxonomy - leave class/determination null
            
            return json.dumps({
                "macroclass": macroclass,
                "taxonomic_class": tax_class,
                "genus": determination,
                "campaign_year": self._extract_year(user_lower),
                "specimen_id": None,
                "collection_code": collection,
                "confidence": 0.85 if macroclass else 0.3
            })
        
        # Regex generation mock - detect collection from filenames
        if "regex" in system_prompt.lower() or "filenames" in user_lower:
            # Check for K-Bue pattern
            if 'k-bue' in user_lower:
                return json.dumps({
                    "master_regex": r"(?P<specimen_id>K-Bue[-\s]?\d+)(?:\s*(?P<plate>[ab]))?",
                    "specimen_id_regex": r"(K-Bue[-\s]?\d+)",
                    "campaign_year_regex": None,
                    "extractable_fields": ["specimen_id", "plate"],
                    "confidence": 0.85
                })
            # Default to LH pattern
            return json.dumps({
                "master_regex": r"(?P<specimen_id>LH[-\s]?\d{3,8})(?:\s*(?P<plate>[ab]))?",
                "specimen_id_regex": r"(LH[-\s]?\d{3,8})",
                "campaign_year_regex": None,
                "extractable_fields": ["specimen_id", "plate"],
                "confidence": 0.85
            })
        
        # File analysis mock
        if "filename" in user_lower:
            return json.dumps({
                "specimen_id": "LH 12345",
                "campaign_year": None,
                "taxonomic_class": None,
                "genus": None,
                "collection_code": collection,
                "confidence": 0.7,
                "is_camera_generated": False
            })
        
        # Default
        return json.dumps({
            "macroclass": None,
            "taxonomic_class": None,
            "genus": None,
            "collection_code": collection,
            "confidence": 0.3
        })
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from text."""
        import re
        match = re.search(r'(20\d{2}|19\d{2})', text)
        if match:
            return int(match.group(1))
        return None


# =============================================================================
# FACTORY & ANALYZER
# =============================================================================

def get_llm_client(provider: str = None) -> BaseLLMClient:
    """
    Factory function to get the appropriate LLM client.
    
    Args:
        provider: "gemini", "github", "local", or "mock".
                  If None, uses config.llm_provider.
        
    Returns:
        BaseLLMClient subclass instance
    """
    provider = provider or config.llm_provider
    
    if provider == 'github':
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed for GitHub Models.")
        return GitHubModelsClient()
    elif provider == 'local':
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed for Local LLM.")
        return LocalLLMClient()
    elif provider == 'mock':
        return MockLLMClient()
    else:
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai not installed for Gemini. Install with: pip install -U google-genai")
        return GeminiClient()


class DirectoryAnalyzer:
    """
    High-level analyzer that processes directories through LLM.
    Handles caching to avoid re-analyzing the same directories.
    """
    
    def __init__(
        self,
        client: BaseLLMClient = None,
        cache_path: Path = None,
        provider: str = None
    ):
        self.client = client
        self._provider = provider
        self.cache_path = cache_path or config.output_base_dir / "directory_analysis_cache.json"
        
        self._cache: Dict[str, dict] = {}
        self._load_cache()
        
        # Stats
        self.analyzed_count = 0
        self.cache_hits = 0
        self.api_calls = 0
        self.errors = 0
    
    def _load_cache(self):
        """Load cache from file."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached analyses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to file."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self._cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def analyze_directory(
        self,
        directory_path: Path,
        sample_filenames: List[str] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> DirectoryAnalysis:
        """
        Analyze a directory using the multi-stage approach.
        """
        path_key = str(directory_path)
        
        # Check cache
        if not force_refresh and path_key in self._cache:
            self.cache_hits += 1
            return DirectoryAnalysis.from_dict(self._cache[path_key])
        
        # Initialize client lazily
        if self.client is None:
            self.client = get_llm_client(self._provider)
        
        # Call LLM
        self.api_calls += 1
        result = self.client.analyze_directory(directory_path, sample_filenames, **kwargs)
        self.analyzed_count += 1
        
        # Cache result
        self._cache[path_key] = result.to_dict()
        self._save_cache()
        
        return result
    
    def analyze_directories(
        self,
        directories: List[Path],
        progress_callback: callable = None
    ) -> Dict[Path, DirectoryAnalysis]:
        """Analyze multiple directories."""
        results = {}
        total = len(directories)
        
        for i, dir_path in enumerate(directories):
            try:
                results[dir_path] = self.analyze_directory(dir_path)
            except Exception as e:
                logger.error(f"Error analyzing {dir_path}: {e}")
                self.errors += 1
                results[dir_path] = DirectoryAnalysis(directory_path=str(dir_path))
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def get_stats(self) -> dict:
        """Get analysis statistics."""
        return {
            'analyzed': self.analyzed_count,
            'cache_hits': self.cache_hits,
            'api_calls': self.api_calls,
            'errors': self.errors,
            'cache_size': len(self._cache),
        }


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Keep the old class name as alias
MockGeminiClient = MockLLMClient


# Convenience function (backwards compatibility)
def analyze_directory(
    directory_path: Path | str,
    sample_filenames: List[str] = None,
    api_key: str = None,
    provider: str = None
) -> DirectoryAnalysis:
    """
    Quick function to analyze a single directory.
    
    Args:
        directory_path: Path to analyze
        sample_filenames: Optional sample filenames
        api_key: API key (for Gemini) or token (for GitHub)
        provider: "gemini", "github", "local", or "mock".
                  If None, uses config.llm_provider
        
    Returns:
        DirectoryAnalysis
    """
    client = get_llm_client(provider)
    return client.analyze_directory(Path(directory_path), sample_filenames)
