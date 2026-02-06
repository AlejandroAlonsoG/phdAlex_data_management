"""
LLM Integration Module v2 for the data ordering tool.
Uses Google Gemini with structured output to extract directory-level metadata.

Key design:
- One LLM call per DIRECTORY (not per file)
- Returns: taxonomic_class, determination, collection_code, campaign_year, specimen_id_regex
- Uses structured output (response_schema) instead of JSON prompting
"""
import os
import time
import logging
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
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
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .config import config


logger = logging.getLogger(__name__)


@dataclass
class DirectoryAnalysis:
    """
    Result of LLM analysis for a directory.
    Contains metadata that applies to ALL files in this directory.
    """
    # Taxonomic classification
    taxonomic_class: Optional[str] = None  # e.g., "Insecta", "Crustacea", "Osteichthyes"
    
    # Most specific taxonomic determination found
    determination: Optional[str] = None  # e.g., "Coleoptera", "Decapoda", "Delclosia"
    
    # Collection/site code
    collection_code: Optional[str] = None  # "LH", "BUE", "MON"
    
    # Campaign year (if detectable from path)
    campaign_year: Optional[int] = None  # e.g., 2018, 2019
    
    # Regex pattern to extract specimen ID from filenames
    specimen_id_regex: Optional[str] = None  # e.g., r"(LH[-\s]?\d{3,8})"
    
    # Confidence in the analysis (0.0 - 1.0)
    confidence: float = 0.0
    
    # The directory path that was analyzed
    directory_path: Optional[str] = None
    
    # Sample filenames used for analysis
    sample_filenames: List[str] = field(default_factory=list)
    
    # Raw response for debugging
    raw_response: Optional[str] = None
    
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
        """
        Apply the regex pattern to extract specimen ID from a filename.
        
        Args:
            filename: The filename to extract from
            
        Returns:
            Extracted specimen ID or None
        """
        if not self.specimen_id_regex:
            return None
        
        try:
            pattern = re.compile(self.specimen_id_regex, re.IGNORECASE)
            match = pattern.search(filename)
            if match:
                # Return the first group if available, else the whole match
                return match.group(1) if match.groups() else match.group(0)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{self.specimen_id_regex}': {e}")
        
        return None


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


# Define the structured output schema for Gemini
DIRECTORY_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "taxonomic_class": {
            "type": "string",
            "description": "Taxonomic class of the fossils (e.g., Insecta, Crustacea, Osteichthyes, Plantae)",
            "nullable": True
        },
        "determination": {
            "type": "string",
            "description": "Most specific taxonomic determination possible (order, family, genus, or species)",
            "nullable": True
        },
        "collection_code": {
            "type": "string",
            "enum": ["LH", "BUE", "MON", "unknown"],
            "description": "Collection site code: LH=Las Hoyas, BUE=Buenache, MON=Montsec"
        },
        "campaign_year": {
            "type": "integer",
            "description": "Year of the campaign/excavation if detectable from the path",
            "nullable": True
        },
        "specimen_id_regex": {
            "type": "string",
            "description": "Python regex pattern with capture group to extract specimen ID from filenames",
            "nullable": True
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this analysis from 0.0 to 1.0"
        }
    },
    "required": ["collection_code", "confidence"]
}


class GeminiClient:
    """
    Client for Google Gemini API with structured output.
    Analyzes directories to extract metadata applicable to all files within.
    """
    
    SYSTEM_PROMPT = """You are an expert paleontologist specializing in Cretaceous fossils from Spanish Konservat-Lagerstätten sites.

You analyze directory paths and sample filenames from a fossil image database to extract:
1. Taxonomic class (Insecta, Crustacea, Osteichthyes, Plantae, Amphibia, Reptilia, Aves, etc.)
2. Most specific taxonomic determination (order, family, genus, or species if identifiable)
3. Collection site code
4. Campaign/excavation year if present in the path
5. A regex pattern to extract specimen IDs from the filenames

COLLECTION SITES:
- Las Hoyas (LH): Prefixes LH, MCCM-LH, MUPA, YCLH, ADL, MDCLM. Cretaceous site from Cuenca, Spain.
- Buenache (BUE): Prefixes K-Bue, Cer-Bue, PB. Another Cretaceous site near Las Hoyas.
- Montsec (MON): Catalan site, different fossil assemblage. Look for references to Montsec, Catalonia, or Lleida.

SPECIMEN ID PATTERNS:
- LH specimens: "LH 12345", "LH-12345", "LH12345" (LH + 3-8 digits)
- MUPA specimens: "MUPA 12345678" (MUPA + digits)
- Buenache: "K-Bue 123", "Cer-Bue 45", "PB 1234" (prefix + digits)
- May have plate indicators: "a", "b" suffix (e.g., "LH 12345 a")

REGEX GUIDELINES:
- Use Python regex syntax
- Always include a capture group for the specimen ID
- Examples:
  - For "LH 12345.jpg": r"(LH[-\\s]?\\d{3,8})"
  - For "MUPA-12345678a.tif": r"(MUPA[-\\s]?\\d+[ab]?)"
  - For "K-Bue 123.jpg": r"(K-Bue[-\\s]?\\d+)"
  - For numeric only "12345.jpg": r"(\\d{3,8})"

Analyze the directory structure and sample filenames to provide accurate metadata."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        requests_per_minute: int = None
    ):
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.environ.get(config.llm_api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {config.llm_api_key_env_var} env var or pass api_key"
            )
        
        self.model_name = model or config.llm_model
        self.rate_limiter = RateLimiter(
            requests_per_minute or config.llm_requests_per_minute
        )
        
        # Configure genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=self.SYSTEM_PROMPT
        )
        
        logger.info(f"Initialized Gemini client with model {self.model_name}")
    
    def analyze_directory(
        self,
        directory_path: Path,
        sample_filenames: List[str] = None,
        max_samples: int = 10,
        max_refinement_attempts: int = 2
    ) -> DirectoryAnalysis:
        """
        Analyze a directory to extract metadata for all its files.
        
        Args:
            directory_path: Path to the directory
            sample_filenames: Optional list of filenames to include
            max_samples: Maximum number of sample filenames to send
            max_refinement_attempts: Max times to ask LLM to refine regex
            
        Returns:
            DirectoryAnalysis with extracted information
        """
        # Get sample filenames if not provided
        if sample_filenames is None:
            try:
                all_files = list(directory_path.iterdir())
                files = [f.name for f in all_files if f.is_file()]
                sample_filenames = files[:max_samples]
            except Exception as e:
                logger.warning(f"Could not list directory {directory_path}: {e}")
                sample_filenames = []
        else:
            sample_filenames = sample_filenames[:max_samples]
        
        # Initial analysis
        result = self._call_llm(directory_path, sample_filenames)
        
        # Validate and refine regex if needed
        if result.specimen_id_regex and sample_filenames:
            result = self._validate_and_refine_regex(
                result, 
                directory_path, 
                sample_filenames,
                max_refinement_attempts
            )
        
        return result
    
    def _call_llm(
        self,
        directory_path: Path,
        sample_filenames: List[str],
        refinement_context: str = None
    ) -> DirectoryAnalysis:
        """Make LLM call for directory analysis."""
        self.rate_limiter.wait_if_needed()
        
        # Build prompt
        prompt = f"""Analyze this directory from a fossil image database:

Directory path: {directory_path}

Sample filenames in this directory:
{chr(10).join(f"- {fn}" for fn in sample_filenames) if sample_filenames else "No samples available"}

Based on the directory path and filenames, determine:
1. What taxonomic class do these fossils belong to?
2. What is the most specific taxonomic determination possible?
3. Which collection site are these from (LH, BUE, MON, or unknown)?
4. Is there a campaign/excavation year visible in the path?
5. What regex pattern would extract the specimen ID from these filenames?"""

        if refinement_context:
            prompt += f"\n\n{refinement_context}"

        try:
            start_time = time.time()
            
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=DIRECTORY_ANALYSIS_SCHEMA
                )
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"Gemini response in {elapsed:.2f}s")
            
            result = self._parse_response(response.text, directory_path, sample_filenames)
            result.raw_response = response.text
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error for {directory_path}: {e}")
            return DirectoryAnalysis(
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
                raw_response=str(e)
            )
    
    def _validate_and_refine_regex(
        self,
        result: DirectoryAnalysis,
        directory_path: Path,
        sample_filenames: List[str],
        max_attempts: int
    ) -> DirectoryAnalysis:
        """
        Validate regex against sample filenames and ask LLM to refine if it fails.
        
        Args:
            result: Current DirectoryAnalysis with regex
            directory_path: Directory being analyzed
            sample_filenames: Sample filenames to test
            max_attempts: Maximum refinement attempts
            
        Returns:
            DirectoryAnalysis with validated/refined regex
        """
        for attempt in range(max_attempts):
            regex = result.specimen_id_regex
            if not regex:
                break
                
            # Test regex against all sample filenames
            failures = []
            successes = []
            
            try:
                pattern = re.compile(regex, re.IGNORECASE)
                for fn in sample_filenames:
                    match = pattern.search(fn)
                    if match:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        successes.append((fn, extracted))
                    else:
                        failures.append(fn)
            except re.error as e:
                # Invalid regex syntax
                logger.warning(f"Invalid regex '{regex}': {e}")
                failures = sample_filenames
            
            # If regex works for most files, accept it
            success_rate = len(successes) / len(sample_filenames) if sample_filenames else 0
            
            if success_rate >= 0.7:  # 70% success threshold
                logger.debug(f"Regex validated: {len(successes)}/{len(sample_filenames)} matches")
                return result
            
            # Regex failed - ask LLM to refine
            if attempt < max_attempts - 1:
                logger.info(f"Regex refinement needed (attempt {attempt + 1}): "
                           f"{len(failures)}/{len(sample_filenames)} failures")
                
                refinement_context = f"""IMPORTANT: Your previous regex pattern failed to extract specimen IDs.

Previous pattern: {regex}

Files where it FAILED to match:
{chr(10).join(f"- {fn}" for fn in failures[:5])}

Files where it SUCCEEDED:
{chr(10).join(f"- {fn} -> {extracted}" for fn, extracted in successes[:3])}

Please provide a CORRECTED regex pattern that will match ALL the sample filenames.
The pattern must include a capture group () for the specimen ID."""

                result = self._call_llm(directory_path, sample_filenames, refinement_context)
            else:
                logger.warning(f"Regex refinement failed after {max_attempts} attempts")
        
        return result
    
    def _parse_response(
        self,
        response_text: str,
        directory_path: Path,
        sample_filenames: List[str]
    ) -> DirectoryAnalysis:
        """Parse structured JSON response from Gemini."""
        try:
            data = json.loads(response_text)
            
            return DirectoryAnalysis(
                taxonomic_class=data.get('taxonomic_class'),
                determination=data.get('determination'),
                collection_code=data.get('collection_code'),
                campaign_year=data.get('campaign_year'),
                specimen_id_regex=data.get('specimen_id_regex'),
                confidence=float(data.get('confidence', 0)),
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response_text}")
            return DirectoryAnalysis(
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
            )


# Try to import OpenAI client for GitHub Models
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class GitHubModelsClient:
    """
    Client for GitHub Models API (OpenAI-compatible).
    Uses models available through GitHub Copilot subscription.
    
    Free models (don't consume monthly tokens):
    - gpt-4o-mini: Fast and capable, good for classification
    - Mistral-small: Good alternative
    - Llama-3.2-11B-Vision-Instruct: Multimodal capable
    - Phi-3-medium-128k-instruct: Microsoft's efficient model
    
    Premium models (consume tokens):
    - gpt-4o, o1-mini, Claude-3.5-Sonnet
    """
    
    SYSTEM_PROMPT = """You are an expert paleontologist specializing in Cretaceous fossils from Spanish Konservat-Lagerstätten sites.

You analyze directory paths and sample filenames from a fossil image database to extract:
1. Taxonomic class (Insecta, Crustacea, Osteichthyes, Plantae, Amphibia, Reptilia, Aves, etc.)
2. Most specific taxonomic determination (order, family, genus, or species if identifiable)
3. Collection site code
4. Campaign/excavation year if present in the path
5. A regex pattern to extract specimen IDs from the filenames

COLLECTION SITES:
- Las Hoyas (LH): Prefixes LH, MCCM-LH, MUPA, YCLH, ADL, MDCLM. Cretaceous site from Cuenca, Spain.
- Buenache (BUE): Prefixes K-Bue, Cer-Bue, PB. Another Cretaceous site near Las Hoyas.
- Montsec (MON): Catalan site, different fossil assemblage. Look for references to Montsec, Catalonia, or Lleida.

SPECIMEN ID PATTERNS:
- LH specimens: "LH 12345", "LH-12345", "LH12345" (LH + 3-8 digits)
- MUPA specimens: "MUPA 12345678" (MUPA + digits)
- Buenache: "K-Bue 123", "Cer-Bue 45", "PB 1234" (prefix + digits)
- May have plate indicators: "a", "b" suffix (e.g., "LH 12345 a")

REGEX GUIDELINES:
- Use Python regex syntax
- Always include a capture group for the specimen ID
- Examples:
  - For "LH 12345.jpg": r"(LH[-\\s]?\\d{3,8})"
  - For "MUPA-12345678a.tif": r"(MUPA[-\\s]?\\d+[ab]?)"

You MUST respond with valid JSON in this exact format:
{
    "taxonomic_class": "string or null",
    "determination": "string or null",
    "collection_code": "LH|BUE|MON|unknown",
    "campaign_year": number or null,
    "specimen_id_regex": "string or null",
    "confidence": number between 0.0 and 1.0
}"""

    def __init__(
        self,
        token: str = None,
        model: str = None,
        endpoint: str = None,
        requests_per_minute: int = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        self.token = token or os.environ.get(config.github_token_env_var)
        if not self.token:
            raise ValueError(
                f"GitHub token required. Set {config.github_token_env_var} env var or pass token.\n"
                "Get your token with: gh auth token"
            )
        
        self.model_name = model or config.github_model
        self.endpoint = endpoint or config.github_models_endpoint
        self.rate_limiter = RateLimiter(
            requests_per_minute or config.llm_requests_per_minute
        )
        
        # Initialize OpenAI client with GitHub Models endpoint
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.token,
        )
        
        logger.info(f"Initialized GitHub Models client with model {self.model_name}")
    
    def analyze_directory(
        self,
        directory_path: Path,
        sample_filenames: List[str] = None,
        max_samples: int = 10,
        max_refinement_attempts: int = 2
    ) -> DirectoryAnalysis:
        """
        Analyze a directory to extract metadata for all its files.
        """
        # Get sample filenames if not provided
        if sample_filenames is None:
            try:
                all_files = list(directory_path.iterdir())
                files = [f.name for f in all_files if f.is_file()]
                sample_filenames = files[:max_samples]
            except Exception as e:
                logger.warning(f"Could not list directory {directory_path}: {e}")
                sample_filenames = []
        else:
            sample_filenames = sample_filenames[:max_samples]
        
        # Initial analysis
        result = self._call_llm(directory_path, sample_filenames)
        
        # Validate and refine regex if needed
        if result.specimen_id_regex and sample_filenames:
            result = self._validate_and_refine_regex(
                result, 
                directory_path, 
                sample_filenames,
                max_refinement_attempts
            )
        
        return result
    
    def _call_llm(
        self,
        directory_path: Path,
        sample_filenames: List[str],
        refinement_context: str = None
    ) -> DirectoryAnalysis:
        """Make LLM call for directory analysis."""
        self.rate_limiter.wait_if_needed()
        
        # Build user prompt
        user_prompt = f"""Analyze this directory from a fossil image database:

Directory path: {directory_path}

Sample filenames in this directory:
{chr(10).join(f"- {fn}" for fn in sample_filenames) if sample_filenames else "No samples available"}

Based on the directory path and filenames, provide your analysis as JSON."""

        if refinement_context:
            user_prompt += f"\n\n{refinement_context}"

        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=512,  # JSON response is small, 512 is plenty
                response_format={"type": "json_object"},  # Enforce JSON output
            )
            
            elapsed = time.time() - start_time
            response_text = response.choices[0].message.content
            logger.info(f"GitHub Models response in {elapsed:.2f}s: {response_text[:200]}...")
            
            result = self._parse_response(response_text, directory_path, sample_filenames)
            result.raw_response = response_text
            
            return result
            
        except Exception as e:
            logger.error(f"GitHub Models API error for {directory_path}: {e}")
            return DirectoryAnalysis(
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
                raw_response=str(e)
            )
    
    def _validate_and_refine_regex(
        self,
        result: DirectoryAnalysis,
        directory_path: Path,
        sample_filenames: List[str],
        max_attempts: int
    ) -> DirectoryAnalysis:
        """Validate regex against sample filenames and ask LLM to refine if it fails."""
        for attempt in range(max_attempts):
            regex = result.specimen_id_regex
            if not regex:
                break
                
            failures = []
            successes = []
            
            try:
                pattern = re.compile(regex, re.IGNORECASE)
                for fn in sample_filenames:
                    match = pattern.search(fn)
                    if match:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        successes.append((fn, extracted))
                    else:
                        failures.append(fn)
            except re.error as e:
                logger.warning(f"Invalid regex '{regex}': {e}")
                failures = sample_filenames
            
            success_rate = len(successes) / len(sample_filenames) if sample_filenames else 0
            
            if success_rate >= 0.7:
                logger.debug(f"Regex validated: {len(successes)}/{len(sample_filenames)} matches")
                return result
            
            if attempt < max_attempts - 1:
                logger.info(f"Regex refinement needed (attempt {attempt + 1})")
                
                refinement_context = f"""IMPORTANT: Your previous regex pattern failed.

Previous pattern: {regex}

Files where it FAILED:
{chr(10).join(f"- {fn}" for fn in failures[:5])}

Files where it SUCCEEDED:
{chr(10).join(f"- {fn} -> {extracted}" for fn, extracted in successes[:3])}

Provide a CORRECTED regex pattern that will match ALL the sample filenames."""

                result = self._call_llm(directory_path, sample_filenames, refinement_context)
            else:
                logger.warning(f"Regex refinement failed after {max_attempts} attempts")
        
        return result
    
    def _parse_response(
        self,
        response_text: str,
        directory_path: Path,
        sample_filenames: List[str]
    ) -> DirectoryAnalysis:
        """Parse JSON response from the model."""
        try:
            # Try to extract JSON from the response (may have markdown code blocks)
            json_text = response_text
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0]
            
            data = json.loads(json_text.strip())
            
            return DirectoryAnalysis(
                taxonomic_class=data.get('taxonomic_class'),
                determination=data.get('determination'),
                collection_code=data.get('collection_code'),
                campaign_year=data.get('campaign_year'),
                specimen_id_regex=data.get('specimen_id_regex'),
                confidence=float(data.get('confidence', 0)),
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
            )
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Full response was: {response_text}")
            # Try to salvage partial data
            return DirectoryAnalysis(
                directory_path=str(directory_path),
                sample_filenames=sample_filenames,
                raw_response=response_text,
            )


def get_llm_client(provider: str = None):
    """
    Factory function to get the appropriate LLM client based on provider.
    
    Args:
        provider: "gemini" or "github". If None, uses config.llm_provider
        
    Returns:
        GeminiClient or GitHubModelsClient
    """
    provider = provider or config.llm_provider
    
    if provider == 'github':
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed for GitHub Models. "
                "Install with: pip install openai"
            )
        return GitHubModelsClient()
    else:
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed for Gemini. "
                "Install with: pip install google-generativeai"
            )
        return GeminiClient()


class DirectoryAnalyzer:
    """
    Processes directories through LLM for metadata extraction.
    Handles caching to avoid re-analyzing the same directories.
    Works with both GeminiClient and GitHubModelsClient.
    """
    
    def __init__(
        self,
        client = None,  # GeminiClient or GitHubModelsClient
        cache_path: Path = None,
        provider: str = None  # "gemini" or "github", uses config if None
    ):
        self.client = client
        self._provider = provider  # Store for lazy initialization
        self.cache_path = cache_path or config.output_base_dir / "directory_analysis_cache.json"
        
        # Cache: directory_path -> DirectoryAnalysis dict
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
                logger.info(f"Loaded {len(self._cache)} cached directory analyses")
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
        force_refresh: bool = False
    ) -> DirectoryAnalysis:
        """
        Analyze a directory, using cache if available.
        
        Args:
            directory_path: Path to analyze
            sample_filenames: Optional list of sample filenames
            force_refresh: If True, ignore cache and re-analyze
            
        Returns:
            DirectoryAnalysis
        """
        path_key = str(directory_path)
        
        # Check cache
        if not force_refresh and path_key in self._cache:
            self.cache_hits += 1
            return DirectoryAnalysis.from_dict(self._cache[path_key])
        
        # Initialize client lazily using factory function
        if self.client is None:
            self.client = get_llm_client(self._provider)
        
        # Call LLM
        self.api_calls += 1
        result = self.client.analyze_directory(directory_path, sample_filenames)
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
        """
        Analyze multiple directories.
        
        Args:
            directories: List of directory paths
            progress_callback: Called with (current, total) after each
            
        Returns:
            Dict mapping directory path to DirectoryAnalysis
        """
        results = {}
        total = len(directories)
        
        for i, dir_path in enumerate(directories):
            try:
                results[dir_path] = self.analyze_directory(dir_path)
            except Exception as e:
                logger.error(f"Error analyzing {dir_path}: {e}")
                self.errors += 1
                results[dir_path] = DirectoryAnalysis(
                    directory_path=str(dir_path)
                )
            
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


class MockGeminiClient:
    """Mock client for testing without actual API calls."""
    
    def analyze_directory(
        self,
        directory_path: Path,
        sample_filenames: List[str] = None,
        max_samples: int = 10
    ) -> DirectoryAnalysis:
        """Mock analysis based on keywords in path."""
        path_lower = str(directory_path).lower()
        samples = sample_filenames or []
        
        # Determine collection
        collection = "unknown"
        if any(x in path_lower for x in ['las hoyas', 'lh-', 'lh/', 'colección lh']):
            collection = "LH"
        elif any(x in path_lower for x in ['buenache', 'bue']):
            collection = "BUE"
        elif any(x in path_lower for x in ['montsec', 'mon', 'catalonia', 'lleida']):
            collection = "MON"
        
        # Determine taxonomic class and determination
        taxonomic_class = None
        determination = None
        
        if 'insecta' in path_lower or 'insecto' in path_lower:
            taxonomic_class = "Insecta"
            if 'coleoptera' in path_lower:
                determination = "Coleoptera"
            elif 'diptera' in path_lower:
                determination = "Diptera"
        elif 'decapoda' in path_lower or 'crustacea' in path_lower:
            taxonomic_class = "Crustacea"
            if 'decapoda' in path_lower:
                determination = "Decapoda"
            if 'delclosia' in path_lower:
                determination = "Delclosia"
        elif 'peces' in path_lower or 'fish' in path_lower or 'osteichthyes' in path_lower:
            taxonomic_class = "Osteichthyes"
        elif 'planta' in path_lower or 'plant' in path_lower:
            taxonomic_class = "Plantae"
        
        # Determine regex pattern based on collection
        regex = None
        if collection == "LH":
            regex = r"(LH[-\s]?\d{3,8})"
        elif collection == "BUE":
            regex = r"(K-Bue[-\s]?\d+|Cer-Bue[-\s]?\d+|PB[-\s]?\d+)"
        
        # Look for year in path
        year = None
        import re
        year_match = re.search(r'(?:campaign|campaña|año)?[-_\s]?(20\d{2}|19\d{2})', path_lower)
        if year_match:
            year = int(year_match.group(1))
        
        return DirectoryAnalysis(
            taxonomic_class=taxonomic_class,
            determination=determination,
            collection_code=collection,
            campaign_year=year,
            specimen_id_regex=regex,
            confidence=0.85 if collection != "unknown" else 0.3,
            directory_path=str(directory_path),
            sample_filenames=samples[:max_samples],
        )


# Convenience function
def analyze_directory(
    directory_path: Path | str,
    sample_filenames: List[str] = None,
    api_key: str = None
) -> DirectoryAnalysis:
    """
    Quick function to analyze a single directory.
    
    Args:
        directory_path: Path to analyze
        sample_filenames: Optional sample filenames
        api_key: Gemini API key (or use env var)
        
    Returns:
        DirectoryAnalysis
    """
    client = GeminiClient(api_key=api_key)
    return client.analyze_directory(Path(directory_path), sample_filenames)
