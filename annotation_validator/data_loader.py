"""
Data loading, path analysis, and session management for annotation validation.

Designed for large datasets:
  - Only anotaciones.xlsx is loaded into memory (lightweight metadata).
  - hashes.xlsx and duplicados_registro.xlsx are indexed once via dicts —
    individual rows are fetched on demand, never as bulk DataFrames kept alive.
  - The O(n²) fallback for duplicate matching is replaced by dict-based
    indexes (path segments, uuid-from-filename).
  - Filter sets (missing fields, has duplicates) are precomputed once so
    sampling is O(1) per candidate.
  - Images are never touched until display time (handled by the UI).
"""

import re
import os
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Set, FrozenSet
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Path Analysis Constants
# ═══════════════════════════════════════════════════════════════════

SPECIMEN_PREFIXES = sorted([
    'MCCM-LH', 'MCLM-LH', 'MCCMLH', 'CER-BUE', 'K-BUE',
    'MCLM', 'MCCM', 'ADR', 'LH', 'PB',
], key=len, reverse=True)

COLLECTION_KEYWORDS = {
    'las hoyas': 'Las Hoyas', 'lashoyas': 'Las Hoyas',
    'hoyas': 'Las Hoyas', 'colección lh': 'Las Hoyas',
    'buenache': 'Buenache', 'montsec': 'Montsec',
}

SOURCE_KEYWORDS = {'mupa': 'MUPA', 'yclh': 'YCLH'}

TAXON_TO_MACROCLASS: Dict[str, str] = {
    'angiosperma': 'Botany', 'bennettitales': 'Botany', 'bryophyta': 'Botany',
    'charophyceae': 'Botany', 'equisetopsida': 'Botany', 'eudicotyledoneae': 'Botany',
    'marchantiopsida': 'Botany', 'pinales': 'Botany', 'pinopsida': 'Botany',
    'planta': 'Botany', 'pteridophyta': 'Botany', 'pterophyta': 'Botany',
    'coleoptera': 'Insects', 'insecta': 'Insects',
    'arachnida': 'Arthropoda', 'branchiopoda': 'Arthropoda', 'crustacea': 'Arthropoda',
    'diplopoda': 'Arthropoda', 'heteropoda': 'Arthropoda', 'malacostraca': 'Arthropoda',
    'ostracoda': 'Arthropoda',
    'bivalvia': 'Mollusca', 'gastropoda': 'Mollusca', 'mollusca': 'Mollusca',
    'clitellata': 'Vermes', 'nematoda': 'Vermes',
    'actinopterygii': 'Pisces', 'chondrichthyes': 'Pisces', 'lepisosteiforme': 'Pisces',
    'osteichthyes': 'Pisces', 'pycnodontiform': 'Pisces', 'sarcopterygii': 'Pisces',
    'amphibia': 'Tetrapoda', 'aves': 'Tetrapoda', 'mammalia': 'Tetrapoda',
    'reptilia': 'Tetrapoda', 'sauropsida': 'Tetrapoda', 'tetrapoda': 'Tetrapoda',
    'testudines': 'Tetrapoda', 'vertebrata': 'Tetrapoda',
    'coprolitos': 'Ichnofossils', 'icnofósil': 'Ichnofossils', 'icnofósiles': 'Ichnofossils',
}
KNOWN_TAXA = set(TAXON_TO_MACROCLASS.keys())
KNOWN_MACROCLASSES = {
    'botany', 'insects', 'arthropoda', 'mollusca', 'vermes',
    'pisces', 'tetrapoda', 'ichnofossils',
}

IMPORTANT_FIELDS = ['specimen_id', 'macroclass_label', 'class_label', 'campaign_year']

# Pipeline filenames: <year>_<stem>_<uuid8>.<ext>
_UUID8_RE = re.compile(r'_([0-9a-fA-F]{8})\.[^.]+$')


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PathClues:
    specimen_ids: List[str] = field(default_factory=list)
    years: List[str] = field(default_factory=list)
    collections: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    taxa: List[str] = field(default_factory=list)
    macroclasses: List[str] = field(default_factory=list)


@dataclass
class DuplicateInfo:
    uuid: str
    original_path: str
    current_path: str
    fields: Dict[str, Any]


@dataclass
class AnnotationSample:
    uuid: str
    fields: Dict[str, Any]
    current_path: str
    original_paths: List[str]
    hash_info: Optional[Dict[str, str]]
    duplicates: List[DuplicateInfo]


# ═══════════════════════════════════════════════════════════════════
# Path Analysis (unchanged logic, called per-sample)
# ═══════════════════════════════════════════════════════════════════

def analyze_paths(paths: List[str]) -> PathClues:
    clues = PathClues()
    seen_specimens: Set[str] = set()
    seen_years: Set[str] = set()
    seen_collections: Set[str] = set()
    seen_sources: Set[str] = set()
    seen_taxa: Set[str] = set()
    seen_macroclasses: Set[str] = set()

    for path_str in paths:
        if not path_str:
            continue
        path_lower = path_str.lower()
        parts = Path(path_str).parts

        for prefix in SPECIMEN_PREFIXES:
            pattern = re.escape(prefix) + r'[\-_\s]?\d+'
            for m in re.finditer(pattern, path_str, re.IGNORECASE):
                sid = m.group(0)
                if sid not in seen_specimens:
                    seen_specimens.add(sid)
                    clues.specimen_ids.append(sid)

        for m in re.finditer(r'(?<!\d)(\d{5,8})(?!\d)', path_str):
            num = m.group(1)
            if num not in seen_specimens:
                seen_specimens.add(num)
                clues.specimen_ids.append(num)

        for m in re.finditer(r'(?<!\d)(19\d{2}|20[0-2]\d)(?!\d)', path_str):
            year = m.group(1)
            if year not in seen_years:
                seen_years.add(year)
                clues.years.append(year)

        for keyword, collection in COLLECTION_KEYWORDS.items():
            if keyword in path_lower and collection not in seen_collections:
                seen_collections.add(collection)
                clues.collections.append(collection)

        for keyword, source in SOURCE_KEYWORDS.items():
            if keyword in path_lower and source not in seen_sources:
                seen_sources.add(source)
                clues.sources.append(source)

        for part in parts:
            part_lower = part.lower().strip()
            if part_lower in KNOWN_TAXA and part_lower not in seen_taxa:
                seen_taxa.add(part_lower)
                clues.taxa.append(part)
                macro = TAXON_TO_MACROCLASS.get(part_lower)
                if macro and macro not in seen_macroclasses:
                    seen_macroclasses.add(macro)
                    clues.macroclasses.append(macro)
            if part_lower in KNOWN_MACROCLASSES and part_lower not in seen_macroclasses:
                seen_macroclasses.add(part_lower.title())
                clues.macroclasses.append(part_lower.title())

    return clues


def compare_field(path_values: List[str], ann_value: Any,
                  case_sensitive: bool = False) -> Tuple[str, str]:
    ann_str = str(ann_value).strip() if ann_value is not None and str(ann_value) != 'nan' else ''
    if not path_values:
        return '—', 'not in path'
    if not ann_str:
        return '⚠', f'in path ({", ".join(path_values[:3])}) but empty in annotation'
    for pv in path_values:
        pv_c = pv if case_sensitive else pv.lower()
        ann_c = ann_str if case_sensitive else ann_str.lower()
        if pv_c == ann_c:
            return '✓', 'match'
        if pv_c in ann_c or ann_c in pv_c:
            return '~', f'partial ({pv} ≈ {ann_str})'
    return '✗', f'mismatch (path: {", ".join(path_values[:3])} ≠ {ann_str})'


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _safe_str(val: Any) -> str:
    """Convert a value to stripped string; None/NaN → ''."""
    if val is None:
        return ''
    s = str(val).strip()
    return '' if s.lower() == 'nan' else s


def _split_paths(path_str: str) -> List[str]:
    """Split a semicolon-separated original_path string into clean parts."""
    return [p.strip() for p in path_str.split(';')
            if p.strip() and p.strip().lower() != 'nan']


def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert a pandas row to a dict, replacing NaN with None."""
    return {col: (val if pd.notna(val) else None) for col, val in row.items()}


def _extract_uuid8(filename: str) -> str:
    """Extract the 8-hex-char UUID from a pipeline filename, or ''."""
    m = _UUID8_RE.search(filename)
    return m.group(1).lower() if m else ''


# ═══════════════════════════════════════════════════════════════════
# DataLoader — lazy, indexed, O(1) per sample
# ═══════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Loads merger output registries with efficient indexing.

    Heavy work is done *once* during ``load()`` using vectorised pandas
    operations and dict comprehensions — no row-by-row iterrows() on the
    full annotation set.  Per-sample lookups are O(1) dict gets.
    """

    SORT_ORDERS = ['path+phash', 'random']

    def __init__(self, output_dir: Path, sort_order: str = 'path+phash'):
        self.output_dir = Path(output_dir)
        self._registries_dir: Optional[Path] = None
        self._sort_order = sort_order if sort_order in self.SORT_ORDERS else 'path+phash'

        # ── Annotation index (lightweight) ──
        # uuid → row-number in _ann_df  (for fast .iloc[])
        self._ann_uuid_to_idx: Dict[str, int] = {}
        self._ann_df: Optional[pd.DataFrame] = None

        # ── Hash index (dict only, df freed after indexing) ──
        # uuid → {md5_hash, phash, file_path}
        self._hash_by_uuid: Dict[str, Dict[str, str]] = {}

        # ── Duplicate index ──
        # main_uuid → [DuplicateInfo, …]
        self._dup_map: Dict[str, List[DuplicateInfo]] = {}
        # set of main uuids that have duplicates (for fast filter)
        self._uuids_with_dups: Set[str] = set()

        # ── Filter sets (precomputed) ──
        self._uuids_missing_fields: Set[str] = set()

        # ── Sampling ──
        self._ordered_uuids: List[str] = []
        self._current_idx: int = 0

        # ── Review tracking ──
        self._reviewed: Dict[str, Dict[str, Any]] = {}

        # ── Path resolution cache ──
        self._path_cache: Dict[str, Optional[Path]] = {}

    # ─── Loading ──────────────────────────────────────────────────

    def load(self) -> Tuple[bool, str]:
        """
        Load & index all registries.  Returns (success, message).

        - anotaciones.xlsx → kept as DataFrame + uuid→idx dict
        - hashes.xlsx → indexed into dict, DataFrame freed
        - duplicados_registro.xlsx → indexed into dict, DataFrame freed
        """
        # Find registries dir
        for name in ['registries', 'registros']:
            candidate = self.output_dir / name
            if candidate.exists():
                self._registries_dir = candidate
                break
        if self._registries_dir is None:
            return False, "No 'registries/' or 'registros/' directory found."

        # ── 1. Annotations ──
        ann_path = self._registries_dir / "anotaciones.xlsx"
        if not ann_path.exists():
            return False, f"anotaciones.xlsx not found in {self._registries_dir}"
        try:
            self._ann_df = pd.read_excel(ann_path)
        except Exception as e:
            return False, f"Error loading anotaciones.xlsx: {e}"
        if 'uuid' not in self._ann_df.columns:
            return False, "No 'uuid' column in anotaciones.xlsx"

        # Build uuid → row-index map
        uuids = self._ann_df['uuid'].astype(str)
        self._ann_uuid_to_idx = {u: i for i, u in enumerate(uuids)}
        n_ann = len(self._ann_df)

        # ── 2. Precompute missing-fields set (vectorised) ──
        self._precompute_missing_fields()

        # ── 3. Hashes (index, then free df) ──
        n_hash = 0
        hash_path = self._registries_dir / "hashes.xlsx"
        if hash_path.exists():
            try:
                hdf = pd.read_excel(hash_path)
                n_hash = len(hdf)
                self._index_hashes(hdf)
                del hdf  # free memory
            except Exception as e:
                logger.warning(f"Could not load hashes.xlsx: {e}")

        # ── 4. Duplicates (index, then free df) ──
        n_dup = 0
        dup_path = self.output_dir / "Duplicados" / "duplicados_registro.xlsx"
        if dup_path.exists():
            try:
                ddf = pd.read_excel(dup_path)
                n_dup = len(ddf)
                self._index_duplicates(ddf)
                del ddf  # free memory
            except Exception as e:
                logger.warning(f"Could not load duplicados_registro.xlsx: {e}")

        # ── 5. Build ordered UUID list ──
        self._ordered_uuids = list(self._ann_uuid_to_idx.keys())
        self._current_idx = 0
        self._history: List[str] = []     # visited UUIDs (for prev navigation)
        self._history_pos: int = -1       # current position within _history
        self.reorder(self._sort_order)

        n_with_dups = len(self._uuids_with_dups)
        return True, (
            f"Loaded {n_ann} annotations, {n_dup} duplicate records, "
            f"{n_hash} hash records.\n"
            f"{n_with_dups} annotations have at least one duplicate."
        )

    # ── Reorder ────────────────────────────────────────────────────

    def reorder(self, sort_order: str):
        """Re-sort the UUID list and reset navigation.

        Supported orders:
          - 'path+phash': group by original-path directory, then by phash
          - 'random': shuffle randomly
        """
        self._sort_order = sort_order

        if sort_order == 'random':
            random.shuffle(self._ordered_uuids)
        else:
            # Default: path + phash
            _no_phash = 'z' * 20

            def _sort_key(uuid: str):
                idx = self._ann_uuid_to_idx[uuid]
                orig = _safe_str(self._ann_df.iloc[idx].get('original_path', ''))
                paths = _split_paths(orig) if orig else []
                if paths:
                    parent = str(Path(paths[0]).parent).lower()
                else:
                    parent = '\xff'
                phash = self._hash_by_uuid.get(uuid, {}).get('phash', _no_phash)
                return (parent, phash)

            self._ordered_uuids.sort(key=_sort_key)

        self._current_idx = 0
        self._history.clear()
        self._history_pos = -1

    # ── Index builders (called once during load) ──────────────────

    def _precompute_missing_fields(self):
        """Vectorised check for rows missing any IMPORTANT_FIELDS."""
        df = self._ann_df
        mask = pd.Series(False, index=df.index)
        for col in IMPORTANT_FIELDS:
            if col in df.columns:
                col_vals = df[col].astype(str).str.strip().str.lower()
                mask |= df[col].isna() | (col_vals == '') | (col_vals == 'nan')
            else:
                # Column doesn't exist at all → every row is "missing"
                mask |= True
        self._uuids_missing_fields = set(
            df.loc[mask, 'uuid'].astype(str).tolist()
        )

    def _index_hashes(self, hdf: pd.DataFrame):
        """Build uuid → hash-info dict from the hashes dataframe."""
        if 'uuid' not in hdf.columns:
            return
        for row in hdf.itertuples(index=False):
            uuid = _safe_str(getattr(row, 'uuid', ''))
            if not uuid:
                continue
            self._hash_by_uuid[uuid] = {
                'md5_hash': _safe_str(getattr(row, 'md5_hash', '')),
                'phash': _safe_str(getattr(row, 'phash', '')),
                'file_path': _safe_str(getattr(row, 'file_path', '')),
            }

    def _index_duplicates(self, ddf: pd.DataFrame):
        """
        Match each duplicate row to its main-record UUID and store in
        ``self._dup_map``.  Uses only O(n+m) dict-based strategies.
        """
        if ddf is None or len(ddf) == 0:
            return

        # ── Build lookup indexes from annotations ──
        # path_segment → main_uuid
        path_to_uuid: Dict[str, str] = {}
        # uuid8 (from filename) → main_uuid
        uuid8_to_uuid: Dict[str, str] = {}

        for row in self._ann_df.itertuples(index=False):
            uuid = _safe_str(getattr(row, 'uuid', ''))
            if not uuid:
                continue

            # Index every semicolon-separated original_path segment
            orig = _safe_str(getattr(row, 'original_path', ''))
            for p in _split_paths(orig):
                path_to_uuid[p] = uuid

            # Index the 8-char uuid from the current_path filename
            cur = _safe_str(getattr(row, 'current_path', ''))
            if cur:
                u8 = _extract_uuid8(cur)
                if u8:
                    uuid8_to_uuid[u8] = uuid

        # ── Match each duplicate ──
        unmatched = 0
        for dup_row in ddf.itertuples(index=False):
            dup_uuid = _safe_str(getattr(dup_row, 'uuid', ''))
            dup_orig = _safe_str(getattr(dup_row, 'original_path', ''))
            dup_current = _safe_str(getattr(dup_row, 'current_path', ''))

            matched_uuid: Optional[str] = None

            # Strategy 1: original_path segment overlap
            for p in _split_paths(dup_orig):
                if p in path_to_uuid:
                    matched_uuid = path_to_uuid[p]
                    break

            # Strategy 2: uuid8 from duplicate's current filename
            if not matched_uuid and dup_current:
                u8 = _extract_uuid8(dup_current)
                if u8 and u8 in uuid8_to_uuid:
                    matched_uuid = uuid8_to_uuid[u8]

            # Strategy 3: uuid8 from duplicate's original path filenames
            if not matched_uuid:
                for p in _split_paths(dup_orig):
                    u8 = _extract_uuid8(p)
                    if u8 and u8 in uuid8_to_uuid:
                        matched_uuid = uuid8_to_uuid[u8]
                        break

            # Build DuplicateInfo (lightweight dict)
            fields = {col: (getattr(dup_row, col) if pd.notna(getattr(dup_row, col, None)) else None)
                      for col in ddf.columns}
            dup_info = DuplicateInfo(
                uuid=dup_uuid,
                original_path=dup_orig,
                current_path=dup_current,
                fields=fields,
            )

            if matched_uuid:
                self._dup_map.setdefault(matched_uuid, []).append(dup_info)
                self._uuids_with_dups.add(matched_uuid)
            else:
                unmatched += 1
                self._dup_map.setdefault('__unmatched__', []).append(dup_info)

        if unmatched:
            logger.warning(f"{unmatched} duplicate records could not be matched to a main record")

    # ─── Path Resolution (cached) ─────────────────────────────────

    def resolve_path(self, path_str: str) -> Optional[Path]:
        """Resolve a file path with caching."""
        if not path_str or path_str in ('nan', 'None', ''):
            return None

        if path_str in self._path_cache:
            return self._path_cache[path_str]

        result = self._resolve_path_uncached(path_str)
        self._path_cache[path_str] = result
        return result

    def _resolve_path_uncached(self, path_str: str) -> Optional[Path]:
        p = Path(path_str)

        # 1. Absolute path
        if p.is_absolute() and p.exists():
            return p

        # 2. Relative to output dir (this is the primary case with relative paths)
        rel = self.output_dir / p
        if rel.exists():
            return rel

        # 3. Try normalising forward slashes to OS separators
        normalised = self.output_dir / Path(path_str.replace('/', os.sep))
        if normalised.exists():
            return normalised

        # 4. Search by filename in common subdirs (one level deep)
        fname = p.name
        search_dirs = [self.output_dir]
        for subdir_name in ('organized', 'Duplicados', 'text_files',
                            'other_files', 'pending_review',
                            'Archivos_Texto', 'Otros_Archivos',
                            'Revision_Manual', 'Casos_Perdidos_Generales'):
            candidate = self.output_dir / subdir_name
            if candidate.is_dir():
                search_dirs.append(candidate)

        for base in search_dirs:
            direct = base / fname
            if direct.exists():
                return direct
            try:
                for child in base.iterdir():
                    if child.is_dir():
                        candidate = child / fname
                        if candidate.exists():
                            return candidate
            except PermissionError:
                continue
        return None

    # ─── Sampling (O(1) per candidate) ────────────────────────────

    def get_sample(self, dup_only: bool = False,
                   missing_only: bool = False) -> Optional[AnnotationSample]:
        """Get the next unreviewed sample (ordered by original path + phash)."""

        # If we went back and there is forward history, replay it first
        if self._history_pos < len(self._history) - 1:
            self._history_pos += 1
            return self._build_sample(self._history[self._history_pos])

        # Otherwise scan for the next unreviewed sample
        n = len(self._ordered_uuids)
        checked = 0

        while checked < n:
            uuid = self._ordered_uuids[self._current_idx]
            self._current_idx = (self._current_idx + 1) % n
            checked += 1

            if uuid in self._reviewed:
                continue
            if dup_only and uuid not in self._uuids_with_dups:
                continue
            if missing_only and uuid not in self._uuids_missing_fields:
                continue

            # Append to history (no truncation needed — we're at the end)
            self._history.append(uuid)
            self._history_pos = len(self._history) - 1

            return self._build_sample(uuid)

        return None

    def get_previous_sample(self) -> Optional[AnnotationSample]:
        """Go back to the previously viewed sample."""
        if self._history_pos <= 0:
            return None
        self._history_pos -= 1
        uuid = self._history[self._history_pos]
        return self._build_sample(uuid)

    def get_sample_by_uuid(self, uuid: str) -> Optional[AnnotationSample]:
        return self._build_sample(uuid)

    def _build_sample(self, uuid: str) -> Optional[AnnotationSample]:
        """Build an AnnotationSample from a UUID using O(1) lookups."""
        idx = self._ann_uuid_to_idx.get(uuid)
        if idx is None:
            return None

        row = self._ann_df.iloc[idx]
        fields = _row_to_dict(row)

        orig_str = _safe_str(row.get('original_path', ''))
        original_paths = _split_paths(orig_str)

        hash_info = self._hash_by_uuid.get(uuid)
        duplicates = self._dup_map.get(uuid, [])

        current_path = _safe_str(row.get('current_path', ''))

        return AnnotationSample(
            uuid=uuid,
            fields=fields,
            current_path=current_path,
            original_paths=original_paths,
            hash_info=hash_info,
            duplicates=duplicates,
        )

    # ─── Validation Tracking ──────────────────────────────────────

    def mark_reviewed(self, uuid: str, verdict: str, notes: str = "",
                      multi_fossil: bool = False):
        self._reviewed[uuid] = {
            'verdict': verdict,
            'notes': notes,
            'multi_fossil': multi_fossil,
            'timestamp': datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._ordered_uuids)
        reviewed = len(self._reviewed)
        verdicts: Dict[str, int] = {}
        for r in self._reviewed.values():
            v = r['verdict']
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            'total': total,
            'reviewed': reviewed,
            'remaining': total - reviewed,
            'verdicts': verdicts,
            'total_duplicates': sum(len(v) for k, v in self._dup_map.items() if k != '__unmatched__'),
            'annotations_with_duplicates': len(self._uuids_with_dups),
            'unmatched_duplicates': len(self._dup_map.get('__unmatched__', [])),
        }

    def save_session(self, filepath: Path = None) -> Path:
        if filepath is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"validation_session_{ts}.xlsx"

        rows = []
        for uuid, review in self._reviewed.items():
            idx = self._ann_uuid_to_idx.get(uuid)
            if idx is None:
                continue
            ann = self._ann_df.iloc[idx]
            rows.append({
                'uuid': uuid,
                'specimen_id': _safe_str(ann.get('specimen_id', '')),
                'original_path': _safe_str(ann.get('original_path', '')),
                'current_path': _safe_str(ann.get('current_path', '')),
                'macroclass_label': _safe_str(ann.get('macroclass_label', '')),
                'class_label': _safe_str(ann.get('class_label', '')),
                'genera_label': _safe_str(ann.get('genera_label', '')),
                'campaign_year': _safe_str(ann.get('campaign_year', '')),
                'fuente': _safe_str(ann.get('fuente', '')),
                'n_duplicates': len(self._dup_map.get(uuid, [])),
                'verdict': review['verdict'],
                'multi_fossil': review.get('multi_fossil', False),
                'notes': review['notes'],
                'reviewed_at': review['timestamp'],
            })

        df = pd.DataFrame(rows)
        df.to_excel(filepath, index=False)
        return filepath
