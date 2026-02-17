"""
Pattern Extraction Module for the data ordering tool.
Extracts specimen IDs, dates, numeric IDs, and taxonomy hints from paths and filenames.
No LLM calls - purely regex and heuristic based.
"""
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
from .config import ALL_SPECIMEN_PREFIXES


class PatternSource(Enum):
    """Where a pattern match came from."""
    FILENAME = "filename"
    PATH = "path"
    REGEX = "regex"
    FALLBACK = "fallback"


@dataclass
class SpecimenIdMatch:
    """Result of specimen ID extraction."""
    specimen_id: str  # The extracted ID (format: cccc-dddddddd-pp)
    prefix: str       # The 4-char prefix (e.g., MUPA, YCLH)
    numeric_part: str # The 8-digit number
    plate: Optional[str]  # The plate indicator (a, b, ab, AB only)
    source: PatternSource
    confidence: float  # 0.0 to 1.0
    raw_match: str    # The original matched text


@dataclass
class NumericIdMatch:
    """Result of numeric ID extraction from filename."""
    numeric_id: str
    is_likely_camera_number: bool  # True if looks like DSC_1234, IMG_0001, etc.
    source: PatternSource
    raw_match: str


@dataclass
class DateMatch:
    """Result of date extraction."""
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    source: PatternSource = PatternSource.FILENAME
    raw_match: str = ""
    
    def to_yyyymmdd(self) -> str:
        """Convert to YYYYMMDD format."""
        if self.month and self.day:
            return f"{self.year:04d}{self.month:02d}{self.day:02d}"
        elif self.month:
            return f"{self.year:04d}{self.month:02d}"
        else:
            return f"{self.year:04d}"
    
    def to_year_string(self) -> str:
        """Just the year as string."""
        return str(self.year)


@dataclass 
class TaxonomyHint:
    """A taxonomy hint found in the path."""
    level: str        # E.g., "class", "order", "family", "genus"
    value: str        # E.g., "Insecta", "Coleoptera"
    source: PatternSource
    path_component: str  # The folder name it came from


@dataclass
class CampaignHint:
    """Campaign year extracted from path."""
    year: int
    source: PatternSource
    raw_match: str


@dataclass
class ExtractionResult:
    """Complete extraction result for a file path."""
    original_path: Path
    specimen_id: Optional[SpecimenIdMatch] = None
    numeric_ids: List[NumericIdMatch] = field(default_factory=list)
    dates: List[DateMatch] = field(default_factory=list)
    taxonomy_hints: List[TaxonomyHint] = field(default_factory=list)
    campaign: Optional[CampaignHint] = None
    
    def has_specimen_id(self) -> bool:
        return self.specimen_id is not None
    
    def get_best_date(self) -> Optional[DateMatch]:
        """Get the most complete date found."""
        if not self.dates:
            return None
        # Sort by completeness (most complete first)
        sorted_dates = sorted(self.dates, 
                             key=lambda d: (d.day is not None, d.month is not None),
                             reverse=True)
        return sorted_dates[0]


class SpecimenIdExtractor:
    """
    Extracts specimen IDs in the format: prefix-numeric-plate
    Where:
        prefix = 2-8 character prefix (LH, MUPA, YCLH, MCCM, MCCM-LH, K-Bue, Cer-Bue)
        numeric = 3-8 digit numeric ID (variable length)
        plate = optional plate indicator, only valid values: a, A, b, B, ab, aB, Ab, AB
    
    Handles real-world variations:
        - LH-15083_a_.jpg, LH-6120.jpg (variable digit lengths)
        - LH 30202 AB.JPG (spaces as separators, double letter plates)
        - LH32560AB.JPG (no separator between prefix and number)
        - MCCM-LH 26452A.JPG (compound prefixes)
        - K-Bue 085.JPG (hyphenated prefixes)
        - Cer-Bue 001a.JPG (longer hyphenated prefixes)
    """
    
    def __init__(self, known_prefixes: List[str] = None):
        """
        Initialize with known prefixes.
        
        Args:
            known_prefixes: List of known prefixes (e.g., ['LH', 'MCLM', 'ADR'])
        """
        # Standard prefixes (may contain hyphens like MCLM-LH)
        # Collections: LH (Las Hoyas), Buenache (K-Bue, Cer-Bue)
        default_prefixes = [
            'LH', 'MCLM', 'MCLM-LH', 'ADR',  # Las Hoyas collection (updated)
            'K-Bue', 'Cer-Bue',        # Buenache collection
        ]
        # If caller didn't provide prefixes, use canonical list from config
        prefixes = known_prefixes if known_prefixes is not None else list(ALL_SPECIMEN_PREFIXES) or default_prefixes
        self.known_prefixes = set(p.upper() for p in prefixes)
        
        self._build_patterns()
    
    # Common camera/device prefixes to exclude from loose matching
    CAMERA_PREFIXES = {'IMG', 'DSC', 'DSCF', 'DSCN', 'DCIM', 'DJI', 'MG', 'PA', 'PC'}
    CAMERA_LONG_PATTERN = re.compile(r'^P1\d{6,}$', re.IGNORECASE)
    
    def _build_patterns(self):
        """Build regex patterns for specimen ID extraction."""
        # Filter out camera prefixes (P1) from pattern matching
        specimen_prefixes = {p for p in self.known_prefixes if p not in ('P1',)}
        
        # Sort prefixes by length (longest first) to avoid partial matches
        sorted_prefixes = sorted(specimen_prefixes, key=len, reverse=True)
        
        # Escape hyphens in prefixes for regex
        escaped_prefixes = [re.escape(p) for p in sorted_prefixes]
        prefix_pattern = '|'.join(escaped_prefixes)
        
        # Full pattern: PREFIX + optional separator + 3-8 digits + optional plate
        # Separator can be: hyphen, underscore, space, or nothing (attached)
        # Plate can only be: a, A, b, B, ab, aB, Ab, AB
        self.full_pattern = re.compile(
            rf'({prefix_pattern})\s*[-_]?\s*(\d{{3,8}})\s*[-_]?\s*([aAbB]{{1,2}})?',
            re.IGNORECASE
        )
        
        # Looser pattern: any 2-4 letters followed by 3-8 digits (not 6+ which is camera)
        # Only matches letter-only prefixes (not compound like MCCM-LH)
        self.loose_pattern = re.compile(
            r'([A-Z]{2,4})\s*[-_]?\s*(\d{3,8})\s*[-_]?\s*([aAbB]{1,2})?',
            re.IGNORECASE
        )
        
        # Pattern for just numeric ID with plate (when prefix might be in path)
        # E.g., "22270A.JPG", "26652B.JPG" - standalone specimen numbers
        self.numeric_only_pattern = re.compile(
            r'(?<![0-9])(\d{3,8})\s*[-_]?\s*([aAbB]{1,2})?(?![0-9])',
            re.IGNORECASE
        )
    
    def extract(self, path: Path) -> Optional[SpecimenIdMatch]:
        """
        Extract specimen ID from a path.
        Tries filename first, then path components.
        
        Args:
            path: Path to extract from
            
        Returns:
            SpecimenIdMatch if found, None otherwise
        """
        # Try filename first with known prefixes
        filename = path.stem  # name without extension
        match = self._try_extract(filename, PatternSource.FILENAME, use_strict=True)
        if match:
            return match
        
        # Try filename with loose pattern
        match = self._try_extract(filename, PatternSource.FILENAME, use_strict=False)
        if match:
            return match
        
        # Try path components
        for part in reversed(path.parts[:-1]):  # Exclude filename
            match = self._try_extract(part, PatternSource.PATH, use_strict=True)
            if match:
                return match
        
        # Try numeric-only in filename (when prefix might be in path)
        prefix_in_path = self._find_prefix_in_path(path)
        if prefix_in_path:
            match = self._try_numeric_only(filename, prefix_in_path, PatternSource.FILENAME)
            if match:
                return match
        
        return None
    
    def _try_extract(self, text: str, source: PatternSource, 
                     use_strict: bool = True) -> Optional[SpecimenIdMatch]:
        """Try to extract specimen ID from text."""
        pattern = self.full_pattern if use_strict else self.loose_pattern
        m = pattern.search(text)
        if m:
            prefix = m.group(1).upper()
            numeric = m.group(2)
            plate = m.group(3) if m.group(3) else None
            
            # Filter out camera prefixes - ALWAYS check, not just for loose pattern
            if prefix in self.CAMERA_PREFIXES:
                return None

            # If using the loose pattern, require the prefix to be one of the
            # known specimen prefixes from configuration. This prevents two-
            # letter accidental matches like "PB 22184b" when "PB" is not
            # a configured prefix.
            if not use_strict and prefix not in self.known_prefixes:
                return None
            
            # Calculate confidence
            confidence = 0.9 if use_strict else 0.7
            if prefix in self.known_prefixes:
                confidence += 0.1
            if plate:
                confidence = min(1.0, confidence + 0.05)
            
            # Normalize the specimen ID format: PREFIX-numeric
            specimen_id = f"{prefix}-{numeric}"
            
            # Normalize plate: convert to lowercase
            if plate:
                plate_normalized = plate.lower()
                specimen_id += f"-{plate_normalized}"
            
            return SpecimenIdMatch(
                specimen_id=specimen_id,
                prefix=prefix,
                numeric_part=numeric,
                plate=plate.lower() if plate else None,
                source=source,
                confidence=confidence,
                raw_match=m.group(0)
            )
        return None
    
    def _find_prefix_in_path(self, path: Path) -> Optional[str]:
        """Find known prefix in path components. Excludes camera prefixes."""
        for part in path.parts:
            part_upper = part.upper()
            for prefix in self.known_prefixes:
                if prefix in part_upper:
                    # Never return P1 - it's always a camera prefix
                    if prefix.upper() not in ('P1',):
                        return prefix
        return None
    
    def _try_numeric_only(self, text: str, prefix: str, 
                          source: PatternSource) -> Optional[SpecimenIdMatch]:
        """Try to extract numeric ID when prefix is in path."""
        m = self.numeric_only_pattern.search(text)
        if m:
            numeric = m.group(1)
            plate = m.group(2).lower() if m.group(2) else None
            

            
            specimen_id = f"{prefix}-{numeric}"
            if plate:
                specimen_id += f"-{plate}"
            
            return SpecimenIdMatch(
                specimen_id=specimen_id,
                prefix=prefix,
                numeric_part=numeric,
                plate=plate,
                source=source,
                confidence=0.6,  # Lower confidence for inferred prefix
                raw_match=m.group(0)
            )
        return None
    
    def add_prefix(self, prefix: str):
        """Add a new known prefix."""
        self.known_prefixes.add(prefix.upper())
        self._build_patterns()


class NumericIdExtractor:
    """
    Extracts numeric IDs from filenames.
    Distinguishes between specimen numbers and camera-generated numbers.
    """
    
    # Camera-related prefixes that indicate camera-generated numbers
    CAMERA_PATTERNS = [
        r'DSC[-_]?\d+',       # DSC_1234, DSC1234, DSC_0495
        r'_DSC[-_]?\d+',      # _DSC variants
        r'DSCN[-_]?\d+',      # Nikon DSCN
        r'IMG[-_]?\d+',       # IMG_1234
        r'DSCF[-_]?\d+',      # Fuji
        r'DCIM[-_]?\d+',
        r'P[A1]\d{6,}',       # PA210066, P1103168 (Olympus, Panasonic with 6+ digits)
        r'PC\d{6,}',          # PC200057 (camera with 6+ digits)
        r'_MG[-_]?\d+',       # Canon
        r'DJI[-_]?\d+',       # DJI drones
    ]
    
    def __init__(self):
        self.camera_pattern = re.compile(
            '|'.join(f'({p})' for p in self.CAMERA_PATTERNS),
            re.IGNORECASE
        )
        # General number pattern (4+ digits)
        self.number_pattern = re.compile(r'(?<![0-9])(\d{4,})(?![0-9])')
    
    def extract_all(self, filename: str) -> List[NumericIdMatch]:
        """
        Extract all numeric IDs from a filename.
        
        Args:
            filename: The filename to analyze (with or without extension)
            
        Returns:
            List of NumericIdMatch objects
        """
        results = []
        
        # First, find camera numbers
        camera_matches = set()
        for m in self.camera_pattern.finditer(filename):
            camera_matches.add(m.group(0))
            # Extract the number part
            nums = re.findall(r'\d+', m.group(0))
            if nums:
                results.append(NumericIdMatch(
                    numeric_id=nums[-1],  # Take last number in pattern
                    is_likely_camera_number=True,
                    source=PatternSource.FILENAME,
                    raw_match=m.group(0)
                ))
        
        # Find other numbers (not part of camera patterns)
        for m in self.number_pattern.finditer(filename):
            # Skip if part of camera pattern
            is_camera = False
            for cm in camera_matches:
                if m.group(1) in cm:
                    is_camera = True
                    break
            
            if not is_camera:
                results.append(NumericIdMatch(
                    numeric_id=m.group(1),
                    is_likely_camera_number=False,
                    source=PatternSource.FILENAME,
                    raw_match=m.group(1)
                ))
        
        return results


class DateExtractor:
    """
    Extracts dates from filenames and paths.
    Handles various formats: YYYYMMDD, YYYY-MM-DD, DD-MM-YYYY, etc.
    """
    
    def __init__(self):
        # Patterns ordered by specificity (most specific first)
        self.patterns = [
            # YYYYMMDD (8 digits together) - most specific
            (r'(?<![0-9])(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])(?![0-9])', 
             'ymd'),
            # YYYY-MM-DD or YYYY/MM/DD (require 2 digits for month and day when separated)
            (r'(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])(?![0-9])', 
             'ymd_sep'),
            # DD-MM-YYYY or DD/MM/YYYY (require 2 digits)
            (r'(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[0-2])[-/](20\d{2})', 
             'dmy_sep'),
            # Just year (4 digits that look like a year)
            (r'(?<![0-9])(20[012]\d)(?![0-9])', 
             'year_only'),
            # Year in folder like "2023 campaign" or "campaign 2023"
            (r'(?:campaign|año|year|temporada)[-_\s]*(20[012]\d)', 
             'campaign_year'),
        ]
        
        self.compiled_patterns = [
            (re.compile(p, re.IGNORECASE), fmt) for p, fmt in self.patterns
        ]
    
    def extract_all(self, path: Path) -> List[DateMatch]:
        """
        Extract all dates from a path.
        
        Args:
            path: Path to analyze
            
        Returns:
            List of DateMatch objects
        """
        results = []
        
        # Try filename first
        filename = path.stem
        filename_dates = self._extract_from_text(filename, PatternSource.FILENAME)
        results.extend(filename_dates)
        
        # Try path components
        for part in path.parts[:-1]:
            path_dates = self._extract_from_text(part, PatternSource.PATH)
            results.extend(path_dates)
        
        # Deduplicate by year-month-day combination
        seen = set()
        unique_results = []
        for r in results:
            key = (r.year, r.month, r.day)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results
    
    def _extract_from_text(self, text: str, source: PatternSource) -> List[DateMatch]:
        """Extract dates from a text string."""
        results = []
        
        for pattern, fmt in self.compiled_patterns:
            for m in pattern.finditer(text):
                date = self._parse_match(m, fmt, source)
                if date:
                    results.append(date)
        
        return results
    
    def _parse_match(self, match, fmt: str, source: PatternSource) -> Optional[DateMatch]:
        """Parse a regex match into a DateMatch."""
        try:
            if fmt == 'ymd':
                return DateMatch(
                    year=int(match.group(1)),
                    month=int(match.group(2)),
                    day=int(match.group(3)),
                    source=source,
                    raw_match=match.group(0)
                )
            elif fmt == 'ymd_sep':
                return DateMatch(
                    year=int(match.group(1)),
                    month=int(match.group(2)),
                    day=int(match.group(3)),
                    source=source,
                    raw_match=match.group(0)
                )
            elif fmt == 'dmy_sep':
                return DateMatch(
                    year=int(match.group(3)),
                    month=int(match.group(2)),
                    day=int(match.group(1)),
                    source=source,
                    raw_match=match.group(0)
                )
            elif fmt in ('year_only', 'campaign_year'):
                year = int(match.group(1))
                # Validate reasonable year range
                if 2000 <= year <= 2030:
                    return DateMatch(
                        year=year,
                        source=source,
                        raw_match=match.group(0)
                    )
        except (ValueError, IndexError):
            pass
        return None


class PathAnalyzer:
    """
    Analyzes paths for taxonomy hints and campaign information.
    """
    
    # Common taxonomy levels and their typical names
    TAXONOMY_HINTS = {
        'phylum': ['Arthropoda', 'Mollusca', 'Chordata', 'Echinodermata', 'Bryozoa', 'Brachiopoda'],
        'class': ['Insecta', 'Arachnida', 'Crustacea', 'Trilobita', 'Gastropoda', 'Bivalvia', 
                  'Cephalopoda', 'Mammalia', 'Reptilia', 'Amphibia', 'Osteichthyes', 'Chondrichthyes'],
        'order': ['Coleoptera', 'Lepidoptera', 'Hymenoptera', 'Diptera', 'Hemiptera', 
                  'Orthoptera', 'Odonata', 'Neuroptera', 'Trichoptera', 'Phasmatodea',
                  'Decapoda', 'Isopoda', 'Ammonoidea', 'Nautiloidea'],
        'family': [],  # Too many to list, will be detected by -idae suffix
        'genus': [],   # Will be detected by position/context
    }
    
    # Campaign-related keywords
    CAMPAIGN_KEYWORDS = ['campaign', 'campaña', 'año', 'year', 'temporada', 'season']
    
    def __init__(self):
        self._build_taxonomy_lookup()
    
    def _build_taxonomy_lookup(self):
        """Build a lookup dict from name to level."""
        self.taxonomy_lookup = {}
        for level, names in self.TAXONOMY_HINTS.items():
            for name in names:
                self.taxonomy_lookup[name.lower()] = level
    
    def analyze(self, path: Path) -> Tuple[List[TaxonomyHint], Optional[CampaignHint]]:
        """
        Analyze a path for taxonomy and campaign hints.
        
        Args:
            path: Path to analyze
            
        Returns:
            Tuple of (taxonomy_hints, campaign_hint)
        """
        taxonomy_hints = []
        campaign_hint = None
        
        for part in path.parts:
            # Check for known taxonomy terms
            part_lower = part.lower()
            if part_lower in self.taxonomy_lookup:
                level = self.taxonomy_lookup[part_lower]
                taxonomy_hints.append(TaxonomyHint(
                    level=level,
                    value=part,  # Keep original case
                    source=PatternSource.PATH,
                    path_component=part
                ))
            
            # Check for family names (-idae suffix)
            if part.endswith('idae') or part.endswith('IDAE'):
                taxonomy_hints.append(TaxonomyHint(
                    level='family',
                    value=part,
                    source=PatternSource.PATH,
                    path_component=part
                ))
            
            # Check for campaign year
            if not campaign_hint:
                campaign_hint = self._check_campaign(part)
        
        return taxonomy_hints, campaign_hint
    
    def _check_campaign(self, part: str) -> Optional[CampaignHint]:
        """Check if a path component contains campaign year info."""
        part_lower = part.lower()
        
        # Look for keyword + year patterns
        for keyword in self.CAMPAIGN_KEYWORDS:
            if keyword in part_lower:
                years = re.findall(r'20[012]\d', part)
                if years:
                    return CampaignHint(
                        year=int(years[0]),
                        source=PatternSource.PATH,
                        raw_match=part
                    )
        
        # Also check for standalone year that looks like a campaign
        if re.match(r'^20[012]\d$', part):
            return CampaignHint(
                year=int(part),
                source=PatternSource.PATH,
                raw_match=part
            )
        
        return None
    
    def find_taxonomy_hints(self, path: Path) -> List[TaxonomyHint]:
        """
        Find taxonomy hints from path components.
        
        This is a convenience method that only returns taxonomy hints
        without analyzing campaign information.
        
        Args:
            path: Path to analyze
            
        Returns:
            List of TaxonomyHint objects found in the path
        """
        hints, _ = self.analyze(path)
        return hints


class PatternExtractor:
    """
    Main pattern extractor that combines all extraction methods.
    """
    
    def __init__(self, known_prefixes: List[str] = None):
        """
        Initialize the pattern extractor.
        
        Args:
            known_prefixes: Known specimen ID prefixes (e.g., ['MUPA', 'YCLH'])
        """
        self.specimen_extractor = SpecimenIdExtractor(known_prefixes)
        self.numeric_extractor = NumericIdExtractor()
        self.date_extractor = DateExtractor()
        self.path_analyzer = PathAnalyzer()
    
    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract all patterns from a path.
        
        Args:
            path: Path to analyze
            
        Returns:
            ExtractionResult with all found patterns
        """
        # Extract specimen ID
        specimen_id = self.specimen_extractor.extract(path)
        
        # Extract numeric IDs from filename
        numeric_ids = self.numeric_extractor.extract_all(path.stem)
        
        # Extract dates
        dates = self.date_extractor.extract_all(path)
        
        # Analyze path for taxonomy and campaign
        taxonomy_hints, campaign = self.path_analyzer.analyze(path)
        
        return ExtractionResult(
            original_path=path,
            specimen_id=specimen_id,
            numeric_ids=numeric_ids,
            dates=dates,
            taxonomy_hints=taxonomy_hints,
            campaign=campaign
        )
    
    def extract_specimen_id_only(self, path: Path) -> Optional[SpecimenIdMatch]:
        """Extract just the specimen ID from a path."""
        return self.specimen_extractor.extract(path)
    
    def add_known_prefix(self, prefix: str):
        """Add a known specimen ID prefix."""
        self.specimen_extractor.add_prefix(prefix)


# Convenience function for quick extraction
def extract_patterns(path: str | Path, known_prefixes: List[str] = None) -> ExtractionResult:
    """
    Convenience function to extract patterns from a path.
    
    Args:
        path: Path string or Path object
        known_prefixes: Known specimen ID prefixes
        
    Returns:
        ExtractionResult with all found patterns
    """
    extractor = PatternExtractor(known_prefixes)
    return extractor.extract(Path(path))
