"""
Merge Output Module for the data ordering tool.
=================================================

Merges a staging directory (produced by the pipeline) into the final output directory.
Handles:
- Copying files and directory structure
- Duplicate detection via MD5/perceptual hashing
- Registry (Excel) merging with conflict resolution
- User prompts for discrepancies

Usage:
    python -m data_ordering.merge_output --staging <staging_dir> [--output <output_dir>]
    
    # Dry run (preview changes)
    python -m data_ordering.merge_output --staging <staging_dir> --dry-run
    
    # Auto-accept all merge decisions
    python -m data_ordering.merge_output --staging <staging_dir> --auto
"""
import argparse
import json
import uuid as _uuid
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime

import pandas as pd

from .image_hasher import ImageHasher, DuplicateType
from .config import IMAGE_EXTENSIONS
from .main_orchestrator import PipelineOrchestrator
from .interaction_manager import (
    InteractionManager, InteractionMode, DecisionType, DecisionOutcome,
    DecisionRequest, DecisionResult,
    create_merge_file_collision_decision,
    create_merge_registry_conflict_decision,
)
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


class MergeConflict:
    """Represents a conflict found during merge."""
    
    def __init__(
        self,
        conflict_type: str,
        staging_path: Path,
        output_path: Path,
        details: Dict[str, Any] = None,
    ):
        self.conflict_type = conflict_type  # 'file_duplicate', 'registry_field', 'file_collision'
        self.staging_path = staging_path
        self.output_path = output_path
        self.details = details or {}
    
    def __repr__(self):
        return f"MergeConflict({self.conflict_type}, {self.staging_path.name})"


class OutputMerger:
    """
    Merges a staging directory into the final output directory.
    
    The staging directory is produced by the pipeline and contains:
    - Organized files in subdirectories
    - registries/ folder with Excel files
    - pipeline_state.json
    - staging_info.json (with final output path)
    - logs/ folder
    """
    
    # Files/dirs that should be merged specially (not just copied)
    REGISTRY_DIR = "registries"
    REGISTRY_FILES = {
        "anotaciones.xlsx",
        "archivos_texto.xlsx",
        "archivos_otros.xlsx",
        "hashes.xlsx",
    }
    # Duplicate registry lives in Duplicados/ folder (not registries/)
    DUPLICATE_REGISTRY = "Duplicados/duplicados_registro.xlsx"
    # Files that are staging-specific metadata and should not be copied
    METADATA_FILES = {
        "pipeline_state.json",
        "staging_info.json",
        "processing_summary.csv",
        "llm_cache.json",
        "deferred_decisions.json",
    }
    METADATA_DIRS = {"logs", "__pycache__"}
    
    def __init__(
        self,
        staging_dir: Path,
        output_dir: Path = None,
        dry_run: bool = False,
        auto_accept: bool = False,
        interaction_mode: str = None,
        image_hasher: ImageHasher = None,
        interaction_manager: InteractionManager = None,
    ):
        """
        Initialize the merger.
        
        Args:
            staging_dir: The staging directory to merge from
            output_dir: The final output directory (auto-detected from staging_info.json if None)
            dry_run: If True, preview changes without modifying anything
            auto_accept: If True, auto-resolve conflicts (skip duplicates, keep existing)
            interaction_mode: 'interactive', 'deferred', 'auto_accept' (overrides auto_accept)
            image_hasher: ImageHasher instance (shared with orchestrator); created if None
            interaction_manager: InteractionManager instance (shared); created if None
        """
        self.staging_dir = Path(staging_dir)
        self.dry_run = dry_run

        # --- Interaction mode ---
        if interaction_mode:
            mode_map = {
                'interactive': InteractionMode.INTERACTIVE,
                'deferred': InteractionMode.DEFERRED,
                'auto_accept': InteractionMode.AUTO_ACCEPT,
                'step_by_step': InteractionMode.STEP_BY_STEP,
            }
            self._interaction_mode = mode_map.get(interaction_mode, InteractionMode.DEFERRED)
        elif auto_accept:
            self._interaction_mode = InteractionMode.AUTO_ACCEPT
        else:
            self._interaction_mode = InteractionMode.INTERACTIVE

        # Load staging info
        self.staging_info = self._load_staging_info()

        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.staging_info and 'final_output_dir' in self.staging_info:
            self.output_dir = Path(self.staging_info['final_output_dir'])
        else:
            raise ValueError(
                "Cannot determine output directory. "
                "Provide --output or ensure staging_info.json exists in staging directory."
            )

        # --- Shared components (same as orchestrator) ---
        self.image_hasher = image_hasher or ImageHasher()
        self.interaction_manager = interaction_manager or InteractionManager(
            mode=self._interaction_mode,
            review_base_dir=self.output_dir / "Manual_Review",
            log_decisions=True,
        )

        # Stats
        self.stats = {
            'files_copied': 0,
            'files_skipped_duplicate': 0,
            'files_skipped_near_duplicate': 0,
            'files_skipped_conflict': 0,
            'directories_created': 0,
            'registry_records_merged': 0,
            'registry_conflicts': 0,
            'hashes_from_registry': 0,
            'hashes_computed': 0,
            'errors': 0,
        }

        self.conflicts: List[MergeConflict] = []

        # Hash caches loaded from hashes.xlsx (populated by _load_hash_registries)
        # Maps normalised file-path string → {'md5_hash': str, 'phash': str}
        self._staging_hashes: Dict[str, Dict[str, str]] = {}
        self._output_hashes: Dict[str, Dict[str, str]] = {}

        # Reverse indexes for the OUTPUT side (populated by _build_output_hash_index)
        # md5 → [file_path, …]; phash → [file_path, …]
        self._output_md5_index: Dict[str, List[str]] = {}
        self._output_phash_index: Dict[str, List[str]] = {}

        # Per-subdirectory choices cache for interactive "apply to subdirectory"
        # mapping: subdir_relative_path -> selected_option (int)
        self._subdir_merge_choices: Dict[str, int] = {}

        # Map copied staging duplicate -> final dest path in output (for registry updates)
        self._copied_duplicates: Dict[str, str] = {}

        # UUIDs detected as duplicates during file merging (shared with registry merge)
        self._file_level_duplicate_uuids: Set[str] = set()

        # Annotation registries (loaded lazily when needed for conflict UI)
        self._staging_anotaciones: Optional[pd.DataFrame] = None
        self._staging_duplicados: Optional[pd.DataFrame] = None
        self._output_anotaciones: Optional[pd.DataFrame] = None
        self._output_duplicados: Optional[pd.DataFrame] = None

    @staticmethod
    def _extract_exif_year_from_paths(paths: List[str]) -> Tuple[Optional[int], Optional[str]]:
        """Try to extract EXIF DateTimeOriginal year from a list of file paths.

        Returns (year:int or None, path_used:str or None)
        Tries paths in order and returns on first successful EXIF year parse.
        """
        if not paths:
            return None, None
        for p in paths:
            try:
                if not p:
                    continue
                pstr = str(p)
                if not os.path.exists(pstr):
                    continue
                with Image.open(pstr) as im:
                    exif = im._getexif() or {}
                if not exif:
                    continue
                # Prefer DateTimeOriginal or DateTime tag
                for tag_id, val in exif.items():
                    name = ExifTags.TAGS.get(tag_id)
                    if name in ('DateTimeOriginal', 'DateTime') and isinstance(val, str):
                        m = __import__('re').search(r"\b(19|20)\d{2}\b", val)
                        if m:
                            return int(m.group(0)), pstr
                # Fallback: search any EXIF string for year pattern
                for val in exif.values():
                    if isinstance(val, str):
                        m = __import__('re').search(r"\b(19|20)\d{2}\b", val)
                        if m:
                            return int(m.group(0)), pstr
            except Exception:
                continue
        return None, None

    def _generate_new_filename(self, file_path: Path, campaign_year: Optional[int] = None, specimen_id: Optional[str] = None) -> str:
        """
        Generate filename similar to the main pipeline.

        Pattern: <year>_<specimen_id_or_stem>_<shortuuid>.<ext>
        """
        file_uuid = str(_uuid.uuid4())[:8]
        year_str = str(campaign_year) if campaign_year else "0000"

        if specimen_id:
            s = str(specimen_id)
            s = re.sub(r'[^\w\-]', '_', s)
            s = re.sub(r'_+', '_', s).strip('_')
            specimen_part = s if s else file_path.stem
        else:
            specimen_part = re.sub(r'[^\w\-]', '_', file_path.stem)
            specimen_part = re.sub(r'_+', '_', specimen_part).strip('_')

        ext = file_path.suffix.lower()
        new_filename = f"{year_str}_{specimen_part}_{file_uuid}{ext}"
        return new_filename

    def _load_staging_info(self) -> Optional[Dict]:
        """Load staging_info.json from the staging directory."""
        info_path = self.staging_dir / "staging_info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _load_hash_registries(self):
        """Load hashes.xlsx from staging and output into in-memory lookup dicts.

        Each dict maps ``file_path`` (as stored in the registry) to
        ``{'md5_hash': ..., 'phash': ...}``.

        After loading, a reverse index is built for the **output** side so
        that every staging file can be checked against *all* output files
        by hash, not just against the file at the same relative path.
        """
        for label, base_dir, target in [
            ('staging', self.staging_dir, '_staging_hashes'),
            ('output', self.output_dir, '_output_hashes'),
        ]:
            reg_path = base_dir / self.REGISTRY_DIR / 'hashes.xlsx'
            if not reg_path.exists():
                logger.debug(f"No hashes.xlsx in {label} directory")
                continue
            try:
                df = pd.read_excel(reg_path)
                cache: Dict[str, Dict[str, str]] = {}
                for _, row in df.iterrows():
                    fp = str(row.get('file_path', '')) if pd.notna(row.get('file_path')) else ''
                    md5 = str(row.get('md5_hash', '')) if pd.notna(row.get('md5_hash')) else ''
                    phash = str(row.get('phash', '')) if pd.notna(row.get('phash')) else ''
                    uuid_val = str(row.get('uuid', '')) if pd.notna(row.get('uuid')) else ''
                    if fp:
                        cache[fp] = {'md5_hash': md5, 'phash': phash, 'uuid': uuid_val}
                setattr(self, target, cache)
                logger.info(f"Loaded {len(cache)} hash entries from {label} hashes.xlsx")
                print(f"  Loaded {len(cache)} pre-computed hashes from {label}")
            except Exception as e:
                logger.warning(f"Failed to load {label} hashes.xlsx: {e}")

        # Build reverse indexes for the output side
        self._build_output_hash_index()

    def _build_output_hash_index(self):
        """Build md5→[paths] and phash→[paths] reverse indexes from output hashes.

        This enables O(1) lookup of any staging file's hash against the
        *entire* output directory — true all-vs-all duplicate detection.
        """
        self._output_md5_index.clear()
        self._output_phash_index.clear()
        for fp, entry in self._output_hashes.items():
            md5 = entry.get('md5_hash', '')
            phash = entry.get('phash', '')
            if md5:
                self._output_md5_index.setdefault(md5, []).append(fp)
            if phash:
                self._output_phash_index.setdefault(phash, []).append(fp)
        logger.info(
            f"Output hash index: {len(self._output_md5_index)} unique MD5s, "
            f"{len(self._output_phash_index)} unique pHashes"
        )

    def _find_hash_match_in_output(
        self, s_md5: str, s_phash: str,
    ) -> Optional[Tuple[str, DuplicateType]]:
        """Check whether a staging file's hashes match ANY file in the output.

        Comparison order (same as the orchestrator's ``ImageHasher.are_duplicates``):

        1. **pHash** — exact match → ``PERCEPTUAL``;
           hamming distance ≤ threshold → ``NEAR``.
        2. **MD5** — exact match → ``EXACT``.

        Returns:
            ``(matched_output_path, DuplicateType)`` if a match is found,
            ``None`` otherwise.
        """
        # 1) Exact phash match
        if s_phash and s_phash in self._output_phash_index:
            return self._output_phash_index[s_phash][0], DuplicateType.PERCEPTUAL

        # 2) Near-duplicate phash (hamming distance ≤ threshold)
        if s_phash:
            for o_phash, o_paths in self._output_phash_index.items():
                distance = self.image_hasher.hamming_distance(s_phash, o_phash)
                if 0 < distance <= self.image_hasher.phash_threshold:
                    return o_paths[0], DuplicateType.NEAR

        # 3) Exact MD5 match
        if s_md5 and s_md5 in self._output_md5_index:
            return self._output_md5_index[s_md5][0], DuplicateType.EXACT

        return None

    def _get_file_hashes(self, file_path: Path, side: str) -> Dict[str, str]:
        """Look up md5/phash for a file from the loaded registry. Does not compute.

        Args:
            file_path: Absolute path to the file.
            side: ``'staging'`` or ``'output'`` — which registry to check.

        Returns:
            ``{'md5_hash': ..., 'phash': ...}`` (values can be empty strings)
        """
        cache = self._staging_hashes if side == 'staging' else self._output_hashes
        
        # First, try lookup with relative path (more robust)
        base_dir = self.staging_dir if side == 'staging' else self.output_dir
        try:
            # Use as_posix() to ensure forward slashes, common in registries
            rel_path_str = file_path.relative_to(base_dir).as_posix()
            if rel_path_str in cache:
                entry = cache.get(rel_path_str, {})
                if entry.get('md5_hash'):
                    self.stats['hashes_from_registry'] += 1
                    return {
                        'md5_hash': str(entry.get('md5_hash', '') or ''),
                        'phash': str(entry.get('phash', '') or ''),
                        'uuid': str(entry.get('uuid', '') or ''),
                    }
        except ValueError:
            pass  # Not a relative path, that's fine

        # Second, try lookup with absolute path as string (original behavior)
        fp_str = str(file_path)
        if fp_str in cache:
            entry = cache.get(fp_str, {})
            if entry.get('md5_hash'):
                self.stats['hashes_from_registry'] += 1
                return {
                    'md5_hash': str(entry.get('md5_hash', '') or ''),
                    'phash': str(entry.get('phash', '') or ''),
                    'uuid': str(entry.get('uuid', '') or ''),
                }

        # Not found in registry — return empty hashes
        return {'md5_hash': '', 'phash': '', 'uuid': ''}

    def _check_missing_hashes(self):
        """Detect staging files that are missing from the hash registry.

        Every mergeable file in the staging directory is checked against
        ``self._staging_hashes``.  If any files lack pre-computed hashes
        the user is prompted to either:

        * **Compute** them on-the-fly (and any other missing ones).
        * **Abort** the merge entirely.

        In ``auto_accept`` mode, missing hashes are computed automatically.

        Raises:
            RuntimeError: If the user chooses to abort.
        """
        missing_files: List[Path] = []
        for file_path in self.staging_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(self.staging_dir)
            if self._should_skip_path(rel_path):
                continue
            # Files inside Duplicados/ are already-identified duplicates whose
            # originals already have hashes.  They were never added to
            # hashes.xlsx by the pipeline, so it's expected (and fine) that
            # they have no entry — skip them from the missing-hash check.
            if rel_path.parts and rel_path.parts[0].lower() == "duplicados":
                continue
            # Only images need hashes for duplicate detection
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            h = self._get_file_hashes(file_path, 'staging')
            if not h.get('md5_hash'):
                missing_files.append(file_path)

        if not missing_files:
            return  # all good

        print(f"\n  WARNING: {len(missing_files)} staging image(s) have no entry in hashes.xlsx.")
        for f in missing_files[:10]:
            print(f"    - {f.relative_to(self.staging_dir)}")
        if len(missing_files) > 10:
            print(f"    ... and {len(missing_files) - 10} more")

        # Decide
        auto = self.interaction_manager.mode in (
            InteractionMode.AUTO_ACCEPT, InteractionMode.DEFERRED,
        )
        if auto:
            choice = 1  # compute automatically
            print("  Auto-mode: computing missing hashes on-the-fly.")
        else:
            print("\n  [1] Compute missing hashes on-the-fly and continue")
            print("  [2] Abort merge")
            raw = input("  Choice [1/2]: ").strip()
            choice = 2 if raw == '2' else 1

        if choice == 2:
            raise RuntimeError(
                f"Merge aborted by user: {len(missing_files)} staging file(s) "
                f"missing from hashes.xlsx. Run the pipeline on the staging "
                f"directory first to generate hashes."
            )

        # Compute hashes for every missing file
        print(f"  Computing hashes for {len(missing_files)} file(s)...")
        for i, file_path in enumerate(missing_files):
            try:
                img_hash = self.image_hasher.hash_image(file_path)
                md5 = img_hash.md5_hash or ''
                phash = str(img_hash.phash) if img_hash.phash else ''
                entry = {'md5_hash': md5, 'phash': phash, 'uuid': ''}

                # Insert into staging cache (both relative and absolute keys)
                fp_abs = str(file_path)
                self._staging_hashes[fp_abs] = entry
                try:
                    rel_key = file_path.relative_to(self.staging_dir).as_posix()
                    self._staging_hashes[rel_key] = entry
                except ValueError:
                    pass

                self.stats['hashes_computed'] += 1
            except Exception as e:
                logger.warning(f"Failed to hash {file_path}: {e}")
                self.stats['errors'] += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(missing_files):
                print(f"    Hashed {i + 1}/{len(missing_files)}")

        print(f"  Done. Computed {self.stats['hashes_computed']} hashes.")

    def _should_skip_path(self, rel_path: Path) -> bool:
        """Check if a path should be skipped during file copy."""
        parts = rel_path.parts
        
        # Skip metadata files at root
        if len(parts) == 1 and parts[0] in self.METADATA_FILES:
            return True
        
        # Skip metadata directories
        if parts[0] in self.METADATA_DIRS:
            return True
        
        # Skip registry files (handled separately)
        if parts[0] == self.REGISTRY_DIR:
            return True
        
        # Skip duplicate registry file (handled in _merge_registries)
        if str(rel_path) == self.DUPLICATE_REGISTRY.replace("/", os.sep):
            return True
        
        return False
    
    def _merge_files(self):
        """Copy non-registry files from staging to output, handling duplicates.

        Duplicate detection is **all-vs-all by hash** — every staging file is
        compared against the entire output directory, not just the file at the
        same relative path.  Hashes come from ``hashes.xlsx`` when available;
        only files missing from the registry are hashed on the fly.

        Comparison order (same as the orchestrator's
        ``ImageHasher.are_duplicates``):

        1. **pHash** — exact match → ``PERCEPTUAL``;
           hamming distance ≤ threshold → ``NEAR``.
        2. **MD5** — exact byte-level match → ``EXACT``.

        Decision flow per staging file:

        * Hash matches an output file (anywhere) → **skip** (duplicate).
        * No hash match but same relative path exists → **file collision**
          → prompt the user via ``InteractionManager``.
        * No hash match and path is free → **copy**.
        """
        print("\n--- Merging Files ---")
        
        # Collect all files in staging (excluding metadata/registries)
        staging_files = []
        for file_path in self.staging_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(self.staging_dir)
            if not self._should_skip_path(rel_path):
                staging_files.append((file_path, rel_path))
        
        total = len(staging_files)
        print(f"  Files to merge: {total}")
        
        for i, (staging_path, rel_path) in enumerate(staging_files):
            # --- 1. Get staging file's hashes (registry first) ---
            s_hashes = self._get_file_hashes(staging_path, 'staging')
            s_md5 = s_hashes['md5_hash']
            s_phash = s_hashes.get('phash', '')
            
            # --- 2. All-vs-all hash match against the entire output ---
            match = self._find_hash_match_in_output(s_md5, s_phash)
            
            if match is not None:
                matched_path, dup_type = match
                logger.info(f"Duplicate detected ({dup_type.value}): {rel_path} ↔ {matched_path}")

                # Track UUID for cross-referencing with registry merge
                s_uuid = s_hashes.get('uuid', '')
                if s_uuid:
                    self._file_level_duplicate_uuids.add(s_uuid)

                # Copy duplicate into output/Duplicados (pipeline-like behavior)
                dup_dir = self.output_dir / 'Duplicados'
                if not self.dry_run:
                    dup_dir.mkdir(parents=True, exist_ok=True)

                # Attempt to extract campaign year from staging or output EXIF
                year, _ = self._extract_exif_year_from_paths([str(staging_path), str(matched_path)])

                # Generate a new filename and ensure uniqueness
                new_name = self._generate_new_filename(staging_path, campaign_year=year, specimen_id=None)
                dest_path = dup_dir / new_name
                counter = 1
                while dest_path.exists():
                    stem = dest_path.stem
                    suffix = dest_path.suffix
                    dest_path = dup_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                try:
                    if not self.dry_run:
                        # Try convert to JPEG first (writes dest_path as given)
                        converted = False
                        # If target extension is not .jpg, try writing .jpg instead
                        if staging_path.suffix.lower() in IMAGE_EXTENSIONS:
                            # prefer .jpg destination
                            jpg_dest = dest_path.with_suffix('.jpg')
                            converted = PipelineOrchestrator._copy_as_jpeg(staging_path, jpg_dest)
                            if converted:
                                final_dest = jpg_dest
                            else:
                                shutil.copy2(staging_path, dest_path)
                                final_dest = dest_path
                        else:
                            # non-image: raw copy
                            shutil.copy2(staging_path, dest_path)
                            final_dest = dest_path
                        # Update in-memory output hashes and indexes so future checks see it
                        self._output_hashes[str(final_dest)] = {'md5_hash': s_md5, 'phash': s_phash}
                        if s_md5:
                            self._output_md5_index.setdefault(s_md5, []).append(str(final_dest))
                        if s_phash:
                            self._output_phash_index.setdefault(s_phash, []).append(str(final_dest))
                        # Record mapping so registry updater can point to the copied duplicate
                        self._copied_duplicates[str(staging_path)] = str(final_dest)
                        logger.info(f"Copied duplicate to: {final_dest}")
                        self.stats['files_copied'] += 1
                except Exception as e:
                    logger.warning(f"Failed to copy duplicate {staging_path} to {dest_path}: {e}")
                # do not treat as a skipped duplicate any more; we've copied it
                continue
            
            # --- 3. No hash match — check for path collision ---
            output_path = self.output_dir / rel_path
            
            if output_path.exists():
                # Same path exists but hashes differ → true file collision
                o_hashes = self._get_file_hashes(output_path, 'output')
                
                conflict = MergeConflict(
                    conflict_type='file_collision',
                    staging_path=staging_path,
                    output_path=output_path,
                    details={
                        'staging_md5': s_md5,
                        'output_md5': o_hashes['md5_hash'],
                        'staging_phash': s_phash,
                        'output_phash': o_hashes.get('phash', ''),
                        'staging_size': staging_path.stat().st_size,
                        'output_size': output_path.stat().st_size,
                    }
                )
                self.conflicts.append(conflict)
                
                # Build enhanced context: original paths and registry fields
                def _load_registries_once():
                    if self._staging_anotaciones is None:
                        ann_path = self.staging_dir / self.REGISTRY_DIR / 'anotaciones.xlsx'
                        dup_path = self.staging_dir / 'Duplicados' / 'duplicados_registro.xlsx'
                        try:
                            self._staging_anotaciones = pd.read_excel(ann_path) if ann_path.exists() else None
                        except Exception:
                            self._staging_anotaciones = None
                        try:
                            self._staging_duplicados = pd.read_excel(dup_path) if dup_path.exists() else None
                        except Exception:
                            self._staging_duplicados = None
                    if self._output_anotaciones is None:
                        ann_path = self.output_dir / self.REGISTRY_DIR / 'anotaciones.xlsx'
                        dup_path = self.output_dir / 'Duplicados' / 'duplicados_registro.xlsx'
                        try:
                            self._output_anotaciones = pd.read_excel(ann_path) if ann_path.exists() else None
                        except Exception:
                            self._output_anotaciones = None
                        try:
                            self._output_duplicados = pd.read_excel(dup_path) if dup_path.exists() else None
                        except Exception:
                            self._output_duplicados = None

                _load_registries_once()

                def _find_row(df: Optional[pd.DataFrame], path: Path) -> Optional[Dict[str, Any]]:
                    if df is None:
                        return None
                    pstr = str(path)
                    name = path.name
                    # Try exact match on common path columns
                    for col in ('original_path', 'current_path', 'file_path'):
                        if col in df.columns:
                            try:
                                matches = df[df[col].astype(str) == pstr]
                            except Exception:
                                matches = pd.DataFrame()
                            if not matches.empty:
                                return matches.iloc[0].to_dict()
                    # Fallback: match by filename ending
                    for col in ('original_path', 'current_path', 'file_path'):
                        if col in df.columns:
                            try:
                                matches = df[df[col].astype(str).str.endswith(name)]
                            except Exception:
                                matches = pd.DataFrame()
                            if not matches.empty:
                                return matches.iloc[0].to_dict()
                    return None

                o_row = _find_row(self._output_anotaciones, output_path) or _find_row(self._output_duplicados, output_path)
                s_row = _find_row(self._staging_anotaciones, staging_path) or _find_row(self._staging_duplicados, staging_path)

                # If we couldn't locate the output registry row by path, try alternate strategies:
                if not o_row:
                    # 1) Try reverse MD5 index (map md5 -> registry file paths)
                    md5_key = o_hashes.get('md5_hash') if isinstance(o_hashes, dict) else None
                    if md5_key:
                        cand_paths = self._output_md5_index.get(md5_key, [])
                        for cand in cand_paths:
                            # cand may be a registry file_path entry (possibly relative)
                            # try to find it in the output registries
                            cand_path_obj = Path(cand)
                            o_row = _find_row(self._output_anotaciones, cand_path_obj) or _find_row(self._output_duplicados, cand_path_obj)
                            if o_row:
                                break
                    # 2) Try matching by UUID if staging row provides one
                    if not o_row and s_row and 'uuid' in s_row:
                        uuid_val = s_row.get('uuid')
                        if uuid_val:
                            for df in (self._output_anotaciones, self._output_duplicados):
                                if df is None:
                                    continue
                                if 'uuid' in df.columns:
                                    try:
                                        matches = df[df['uuid'].astype(str) == str(uuid_val)]
                                    except Exception:
                                        matches = pd.DataFrame()
                                    if not matches.empty:
                                        o_row = matches.iloc[0].to_dict()
                                        break

                def _compact_fields(row: Optional[Dict[str, Any]]) -> Dict[str, str]:
                    if not row:
                        return {}
                    out = {}
                    for k, v in row.items():
                        try:
                            if pd.isna(v):
                                continue
                        except Exception:
                            pass
                        if isinstance(v, (int, float)):
                            out[k] = str(v)
                        else:
                            s = str(v)
                            out[k] = s if len(s) <= 120 else s[:117] + '...'
                    return out

                o_fields = _compact_fields(o_row)
                s_fields = _compact_fields(s_row)

                # Build differences for quick glance
                diffs = {}
                all_keys = set(o_fields.keys()) | set(s_fields.keys())
                for k in sorted(all_keys):
                    ov = o_fields.get(k, '')
                    sv = s_fields.get(k, '')
                    if ov != sv:
                        diffs[k] = {'output': ov, 'staging': sv}

                context_info = {
                    'existing_original_path': o_row.get('original_path') if o_row and 'original_path' in o_row else '',
                    'staging_original_path': s_row.get('original_path') if s_row and 'original_path' in s_row else '',
                    'existing_registry_fields': json.dumps(o_fields, ensure_ascii=False) if o_fields else '',
                    'staging_registry_fields': json.dumps(s_fields, ensure_ascii=False) if s_fields else '',
                }

                if diffs:
                    # Build a pretty comparison table with original paths prominently shown
                    match_display = output_path.name
                    o_orig = o_row.get('original_path') if o_row and 'original_path' in o_row else ''
                    s_orig = s_row.get('original_path') if s_row and 'original_path' in s_row else ''
                    ctx_info_lines = []
                    if o_orig or s_orig:
                        ctx_info_lines.append(f"original_path (existing): {o_orig or '(empty)'}")
                        ctx_info_lines.append(f"original_path (staging):  {s_orig or '(empty)'}")
                    
                    # Add EXIF year info if campaign_year is in the differences
                    if 'campaign_year' in diffs:
                        o_year, o_from = self._extract_exif_year_from_paths([str(output_path)])
                        s_year, s_from = self._extract_exif_year_from_paths([str(staging_path)])
                        if o_year or s_year:
                            ctx_info_lines.append("")  # blank line for separation
                            if o_year:
                                ctx_info_lines.append(f"EXIF year (existing): {o_year}")
                            else:
                                ctx_info_lines.append("EXIF year (existing): (no EXIF found)")
                            if s_year:
                                ctx_info_lines.append(f"EXIF year (staging):  {s_year}")
                            else:
                                ctx_info_lines.append("EXIF year (staging):  (no EXIF found)")
                    # Add match reason and relevant hash/pHash info for debugging
                    try:
                        ctx_info_lines.append("")
                        ctx_info_lines.append("Match reason: file collision (same path exists with differing content)")
                        # Indicate whether hashes came from registries or were computed
                        s_src = 'computed'
                        o_src = 'computed'
                        try:
                            sp = str(staging_path)
                            if sp in self._staging_hashes and self._staging_hashes[sp].get('md5_hash'):
                                s_src = 'staging registry'
                        except Exception:
                            pass
                        try:
                            op = str(output_path)
                            if op in self._output_hashes and self._output_hashes[op].get('md5_hash'):
                                o_src = 'output registry'
                        except Exception:
                            pass
                        ctx_info_lines.append(f"staging_md5 ({s_src}): {s_md5}")
                        ctx_info_lines.append(f"output_md5 ({o_src}): {o_hashes.get('md5_hash', '')}")
                        o_ph = o_hashes.get('phash', '') if isinstance(o_hashes, dict) else ''
                        if o_ph:
                            ctx_info_lines.append(f"pHash (existing): {o_ph}")
                        if s_phash:
                            ctx_info_lines.append(f"pHash (staging):  {s_phash}")
                    except Exception:
                        pass
                    
                    table = self._build_comparison_table(diffs, match_display, context_info=ctx_info_lines if ctx_info_lines else None)
                    extra_message = table
                else:
                    # Always show original paths even if no differing registry fields
                    o_orig = o_row.get('original_path') if o_row and 'original_path' in o_row else ''
                    s_orig = s_row.get('original_path') if s_row and 'original_path' in s_row else ''
                    if o_orig or s_orig:
                        extra_message = (
                            f"original_path (existing): {o_orig or '(empty)'}\n"
                            f"original_path (staging):  {s_orig or '(empty)'}"
                        )
                    else:
                        extra_message = '(no registry data found)'

                # Determine subdirectory key from original_path in annotations
                # (parsing ';'-separated values written by the orchestrator)
                subdir_key = ''
                s_orig_path = s_row.get('original_path', '') if s_row else ''
                o_orig_path = o_row.get('original_path', '') if o_row else ''
                orig_for_key = str(s_orig_path or o_orig_path or '')
                if orig_for_key:
                    subdir_key = self._extract_subdir_key(orig_for_key)
                if not subdir_key:
                    subdir_key = str(rel_path.parent)  # fallback to current dir

                # If user has already chosen an "apply to subdirectory" action,
                # use it without prompting again.
                cached_choice = self._subdir_merge_choices.get(subdir_key)
                if cached_choice is not None:
                    choice = cached_choice
                else:
                    decision_request = create_merge_file_collision_decision(
                        staging_path=staging_path,
                        output_path=output_path,
                        staging_md5=s_md5,
                        output_md5=o_hashes['md5_hash'],
                        staging_size=staging_path.stat().st_size,
                        output_size=output_path.stat().st_size,
                        context_info=context_info,
                        extra_message=extra_message,
                    )
                    result = self.interaction_manager.request_decision(decision_request)

                    # Handle defer
                    if result.outcome == DecisionOutcome.DEFER:
                        self.stats['files_skipped_conflict'] += 1
                        continue

                    # If user asked to apply this choice to the whole subdirectory,
                    # store it and reuse for subsequent files in the same subdir.
                    if result.outcome == DecisionOutcome.APPLY_TO_SUBDIRECTORY:
                        sel = result.selected_option if result.selected_option is not None else 0
                        self._subdir_merge_choices[subdir_key] = sel
                        choice = sel
                    else:
                        choice = result.selected_option if result.selected_option is not None else 0

                # Execute the chosen action
                if choice == 0:
                    # Keep existing
                    self.stats['files_skipped_conflict'] += 1
                    continue
                elif choice == 1:
                    # Overwrite
                    if not self.dry_run:
                        shutil.copy2(staging_path, output_path)
                    self.stats['files_copied'] += 1
                elif choice == 2:
                    # Keep both — rename staging file
                    stem = output_path.stem
                    suffix = output_path.suffix
                    counter = 1
                    new_path = output_path.parent / f"{stem}_merged_{counter}{suffix}"
                    while new_path.exists():
                        counter += 1
                        new_path = output_path.parent / f"{stem}_merged_{counter}{suffix}"
                    if not self.dry_run:
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(staging_path, new_path)
                    self.stats['files_copied'] += 1
            else:
                # --- 4. No hash match, path is free → copy ---
                if not self.dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staging_path, output_path)
                self.stats['files_copied'] += 1
            
            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1}/{total}")
        
        print(f"  Files copied: {self.stats['files_copied']}")
        print(f"  Duplicates skipped: {self.stats['files_skipped_duplicate']}")
        if self.stats['files_skipped_near_duplicate']:
            print(f"    (of which near-duplicates by phash: {self.stats['files_skipped_near_duplicate']})")
        print(f"  Conflicts resolved: {self.stats['files_skipped_conflict']}")
        print(f"  Hashes from registry: {self.stats['hashes_from_registry']}")
        print(f"  Hashes computed on-the-fly: {self.stats['hashes_computed']}")

    def _merge_registries(self):
        """Merge Excel registry files from staging into output.

        The *anotaciones* and *hashes* registries are merged together in a
        single cross-referenced pass: for each staging image we look for a
        match by hash (pHash first, then MD5) in the output, then reconcile
        the annotation fields.

        The remaining registries (archivos_texto, archivos_otros,
        duplicados_registro) are merged independently by original_path.
        """
        print("\n--- Merging Registries ---")

        staging_reg_dir = self.staging_dir / self.REGISTRY_DIR
        output_reg_dir = self.output_dir / self.REGISTRY_DIR

        if not staging_reg_dir.exists():
            print("  No registries in staging directory")
            return

        if not self.dry_run:
            output_reg_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Cross-referenced merge of annotations + hashes ───────
        self._merge_annotations_and_hashes(staging_reg_dir, output_reg_dir)

        # ── 2. Simple registries ────────────────────────────────────
        for registry_file in ("archivos_texto.xlsx", "archivos_otros.xlsx"):
            staging_reg = staging_reg_dir / registry_file
            output_reg = output_reg_dir / registry_file

            if not staging_reg.exists():
                continue

            print(f"\n  Merging: {registry_file}")

            if not output_reg.exists():
                if not self.dry_run:
                    shutil.copy2(staging_reg, output_reg)
                staging_df = pd.read_excel(staging_reg)
                self.stats['registry_records_merged'] += len(staging_df)
                print(f"    Copied {len(staging_df)} records (new registry)")
                continue

            try:
                staging_df = pd.read_excel(staging_reg)
                output_df = pd.read_excel(output_reg)
                merge_result = self._merge_dataframes(
                    staging_df, output_df,
                    match_cols=['original_path'],
                    registry_name=registry_file,
                )
                if merge_result is not None and not self.dry_run:
                    merge_result.to_excel(output_reg, index=False)
            except Exception as e:
                logger.error(f"Failed to merge {registry_file}: {e}")
                self.stats['errors'] += 1
                print(f"    ERROR: {e}")

        # ── 3. Duplicate registry (Duplicados/) ─────────────────────
        staging_dup_reg = self.staging_dir / self.DUPLICATE_REGISTRY
        output_dup_reg = self.output_dir / self.DUPLICATE_REGISTRY
        if staging_dup_reg.exists():
            print(f"\n  Merging: {self.DUPLICATE_REGISTRY}")
            try:
                if not output_dup_reg.exists():
                    if not self.dry_run:
                        output_dup_reg.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(staging_dup_reg, output_dup_reg)
                    staging_df = pd.read_excel(staging_dup_reg)
                    self.stats['registry_records_merged'] += len(staging_df)
                    print(f"    Copied {len(staging_df)} duplicate records (new registry)")
                else:
                    # Staging duplicate registry exists and output duplicate registry
                    # also exists. For duplicates coming from different sources we
                    # consider it safe to append all rows without attempting any
                    # matching or conflict resolution (UUIDs and original paths
                    # are expected to be distinct). Simply concatenate the
                    # staging records to the output registry.
                    staging_df = pd.read_excel(staging_dup_reg)
                    try:
                        output_df = pd.read_excel(output_dup_reg)
                    except Exception:
                        output_df = pd.DataFrame()

                    if len(staging_df):
                        combined = pd.concat([output_df, staging_df], ignore_index=True)
                        if not self.dry_run:
                            output_dup_reg.parent.mkdir(parents=True, exist_ok=True)
                            combined.to_excel(output_dup_reg, index=False)
                        self.stats['registry_records_merged'] += len(staging_df)
                        print(f"    Appended {len(staging_df)} duplicate records (staging -> output)")
            except Exception as e:
                logger.error(f"Failed to merge duplicate registry: {e}")
                self.stats['errors'] += 1
                print(f"    ERROR: {e}")
    
    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_specimen_id(value: str) -> str:
        """Normalise a specimen_id for comparison.

        Strips separators (-, _, spaces), lowercases, and removes
        surrounding whitespace so that e.g. 'LH-22270-a' matches
        'lh22270a'.
        """
        if not value:
            return ''
        return re.sub(r'[\s\-_]+', '', value.strip()).lower()

    @staticmethod
    def _merge_original_paths(existing: str, incoming: str) -> str:
        """Combine two original_path values into a deduplicated ';'-separated list."""
        parts: list[str] = []
        for raw in (existing, incoming):
            if raw:
                for p in raw.split(';'):
                    p = p.strip()
                    if p and p not in parts:
                        parts.append(p)
        return '; '.join(parts)

    @staticmethod
    def _extract_subdir_key(original_path_str: str) -> str:
        """Extract a subdirectory key from an original_path value.

        The original_path may contain multiple ';'-separated paths
        (as written by the main orchestrator when duplicates merge).
        Returns the parent directory of the FIRST path.
        """
        if not original_path_str:
            return ''
        paths = [p.strip() for p in original_path_str.split(';') if p.strip()]
        if not paths:
            return ''
        try:
            return str(Path(paths[0]).parent)
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Unified cross-referenced merge (annotations + hashes)
    # ------------------------------------------------------------------

    def _merge_annotations_and_hashes(
        self,
        staging_reg_dir: Path,
        output_reg_dir: Path,
    ):
        """Merge anotaciones.xlsx and hashes.xlsx together, handling duplicates and conflicts as described."""
        import pandas as pd
        from collections import defaultdict

        staging_ann_path = staging_reg_dir / "anotaciones.xlsx"
        staging_hash_path = staging_reg_dir / "hashes.xlsx"
        output_ann_path = output_reg_dir / "anotaciones.xlsx"
        output_hash_path = output_reg_dir / "hashes.xlsx"
        output_dup_reg_path = self.output_dir / "Duplicados" / "duplicados_registro.xlsx"

        # Load data
        s_ann = pd.read_excel(staging_ann_path) if staging_ann_path.exists() else pd.DataFrame()
        s_hash = pd.read_excel(staging_hash_path) if staging_hash_path.exists() else pd.DataFrame()
        o_ann = pd.read_excel(output_ann_path) if output_ann_path.exists() else pd.DataFrame()
        o_hash = pd.read_excel(output_hash_path) if output_hash_path.exists() else pd.DataFrame()
        o_dup = pd.read_excel(output_dup_reg_path) if output_dup_reg_path.exists() else pd.DataFrame()

        if s_ann.empty or s_hash.empty:
            print("  Staging anotaciones.xlsx or hashes.xlsx is missing/empty. Skipping merge.")
            return

        # Prepare lookups
        for df in [s_ann, s_hash, o_ann, o_hash]:
            if 'uuid' in df.columns:
                df['uuid'] = df['uuid'].astype(str)

        s_ann_by_uuid = s_ann.set_index('uuid').to_dict('index') if not s_ann.empty and 'uuid' in s_ann.columns else {}
        o_ann_by_uuid = o_ann.set_index('uuid').to_dict('index') if not o_ann.empty and 'uuid' in o_ann.columns else {}

        # Build hash indexes for output
        o_hash_by_md5 = defaultdict(list)
        o_hash_by_phash = defaultdict(list)
        o_hash_phash_map = {}
        o_hash_md5_map = {}
        phash_threshold = self.image_hasher.phash_threshold
        for _, row in o_hash.iterrows():
            md5 = str(row.get('md5_hash', ''))
            phash = str(row.get('phash', ''))
            uuid = str(row.get('uuid', ''))
            if md5:
                o_hash_by_md5[md5].append(uuid)
                o_hash_md5_map[uuid] = md5
            if phash:
                o_hash_by_phash[phash].append(uuid)
                o_hash_phash_map[uuid] = phash

        # Prepare new output annotation and hash DataFrames
        output_ann_copy = o_ann.copy()
        output_hash_copy = o_hash.copy()
        output_dup_copy = o_dup.copy()

        # Track which staging UUIDs are duplicates
        duplicate_uuids = set()
        processed_output_uuids = set()
        rows_auto_merged = 0
        conflicts_resolved = 0
        new_ann_rows = []
        new_hash_rows = []
        new_dup_rows = []

        for _, s_hash_row in s_hash.iterrows():
            s_uuid = str(s_hash_row.get('uuid', ''))
            s_phash = str(s_hash_row.get('phash', ''))
            s_md5 = str(s_hash_row.get('md5_hash', ''))
            s_ann_row = s_ann_by_uuid.get(s_uuid)
            if not s_uuid or not s_ann_row:
                continue

            # 1. Try exact pHash match
            match_uuid = None
            match_type = None
            if s_phash and s_phash in o_hash_by_phash:
                match_uuid = o_hash_by_phash[s_phash][0]
                match_type = 'pHash'
            # 2. Try near-duplicate pHash (threshold)
            elif s_phash:
                for o_phash, uuids in o_hash_by_phash.items():
                    if o_phash and s_phash and o_phash != s_phash:
                        dist = self.image_hasher.hamming_distance(s_phash, o_phash)
                        if 0 < dist <= phash_threshold:
                            match_uuid = uuids[0]
                            match_type = f'pHash~{dist}'
                            break
            # 3. Try exact MD5 match
            if not match_uuid and s_md5 and s_md5 in o_hash_by_md5:
                match_uuid = o_hash_by_md5[s_md5][0]
                match_type = 'md5'

            if match_uuid and match_uuid in o_ann_by_uuid:
                # DUPLICATE: add to duplicados_registro, do NOT add to hashes.xlsx
                duplicate_uuids.add(s_uuid)
                s_row = pd.Series(s_ann_row)
                o_row = pd.Series(o_ann_by_uuid[match_uuid])
                # Always accumulate original_path
                if 'original_path' in s_row.index and 'original_path' in o_row.index:
                    s_orig = str(s_row.get('original_path', ''))
                    o_orig = str(o_row.get('original_path', ''))
                    merged_paths = self._merge_original_paths(o_orig, s_orig)
                    if merged_paths != o_orig:
                        output_ann_copy.loc[output_ann_copy['uuid'] == match_uuid, 'original_path'] = merged_paths

                # Field-by-field comparison for annotation
                discrepancies = {}
                skip_cols = {'created_at', 'updated_at', 'timestamp', 'uuid', 'id', 'current_path', 'original_path'}
                for col in s_row.index:
                    if col in skip_cols or col not in o_row.index:
                        continue
                    s_val = s_row.get(col)
                    o_val = o_row.get(col)
                    s_str = str(s_val) if pd.notna(s_val) else ''
                    o_str = str(o_val) if pd.notna(o_val) else ''
                    if s_str == o_str:
                        continue
                    if not o_str and s_str:
                        if not self.dry_run:
                            output_ann_copy.loc[output_ann_copy['uuid'] == match_uuid, col] = s_val
                        rows_auto_merged += 1
                    elif o_str and not s_str:
                        pass
                    else:
                        discrepancies[col] = {'staging': s_str, 'output': o_str}

                # Prompt for real conflicts
                if discrepancies:
                    ctx_lines = [
                        f"specimen_id (existing): {o_row.get('specimen_id', '(no id)')}",
                        f"specimen_id (staging):  {s_row.get('specimen_id', '(no id)')}",
                        f"original_path (existing): {o_row.get('original_path', '')}",
                        f"original_path (staging):  {s_row.get('original_path', '')}",
                        "",
                        f"Match reason: {match_type} match",
                    ]
                    # When campaign_year is disputed, show EXIF years to help decide
                    if 'campaign_year' in discrepancies:
                        o_cur = str(o_row.get('current_path', '')) if pd.notna(o_row.get('current_path')) else ''
                        s_cur = str(s_row.get('current_path', '')) if pd.notna(s_row.get('current_path')) else ''
                        o_exif_year, _ = self._extract_exif_year_from_paths([o_cur])
                        s_exif_year, _ = self._extract_exif_year_from_paths([s_cur])
                        ctx_lines.append("")
                        ctx_lines.append(f"EXIF year (existing): {o_exif_year if o_exif_year else 'not available'}")
                        ctx_lines.append(f"EXIF year (staging):  {s_exif_year if s_exif_year else 'not available'}")
                    table = self._build_comparison_table(discrepancies, f"{match_type} match", context_info=ctx_lines)
                    # Subdir key from original_path
                    orig_path = s_row.get('original_path', '') or o_row.get('original_path', '')
                    subdir_key = self._extract_subdir_key(orig_path)
                    cached = self._subdir_merge_choices.get(subdir_key)
                    choice = None
                    if cached is not None:
                        choice = cached
                    else:
                        decision_request = create_merge_registry_conflict_decision(
                            registry_name='anotaciones.xlsx',
                            match_label=f"Merge conflict ({match_type})",
                            comparison_table=table,
                            context_info={}
                        )
                        result = self.interaction_manager.request_decision(decision_request)
                        if result.outcome == DecisionOutcome.APPLY_TO_SUBDIRECTORY:
                            sel = result.selected_option if result.selected_option is not None else 0
                            self._subdir_merge_choices[subdir_key] = sel
                            choice = sel
                        else:
                            choice = result.selected_option if result.selected_option is not None else 0
                    if choice == 1:
                        for col in discrepancies:
                            if not self.dry_run:
                                output_ann_copy.loc[output_ann_copy['uuid'] == match_uuid, col] = s_row[col]
                        conflicts_resolved += 1
                        self.stats['registry_conflicts'] += 1

                # Add the duplicate row to duplicados_registro
                # Re-inject uuid (set_index('uuid') removed it from the dict)
                dup_dict = dict(s_ann_row)
                dup_dict['uuid'] = s_uuid
                new_dup_rows.append(pd.Series(dup_dict))
                continue

            # NOT DUPLICATE: add to new_ann_rows and new_hash_rows
            # Re-inject uuid (set_index('uuid') removed it from the dict)
            ann_dict = dict(s_ann_row)
            ann_dict['uuid'] = s_uuid
            new_ann_rows.append(pd.Series(ann_dict))
            new_hash_rows.append(s_hash_row)

        # Cross-check: compare registry-level duplicates with file-level duplicates
        if self._file_level_duplicate_uuids:
            only_file = self._file_level_duplicate_uuids - duplicate_uuids
            only_reg = duplicate_uuids - self._file_level_duplicate_uuids
            if only_file:
                logger.warning(
                    f"File-level detected {len(only_file)} duplicate UUID(s) "
                    f"not caught by registry merge: {list(only_file)[:5]}"
                )
            if only_reg:
                logger.warning(
                    f"Registry merge detected {len(only_reg)} duplicate UUID(s) "
                    f"not caught by file-level merge: {list(only_reg)[:5]}"
                )
            # Merge both sets so removals cover both passes
            duplicate_uuids = duplicate_uuids | self._file_level_duplicate_uuids

        # Append new annotation and hash records (non-duplicates only)
        if new_ann_rows:
            output_ann_copy = pd.concat([output_ann_copy, pd.DataFrame(new_ann_rows)], ignore_index=True)
        if new_hash_rows:
            output_hash_copy = pd.concat([output_hash_copy, pd.DataFrame(new_hash_rows)], ignore_index=True)
            # Remove any hash rows whose uuid is in duplicate_uuids
            output_hash_copy = output_hash_copy[~output_hash_copy['uuid'].isin(list(duplicate_uuids))]

        # Append new duplicates to duplicados_registro
        if new_dup_rows:
            output_dup_copy = pd.concat([output_dup_copy, pd.DataFrame(new_dup_rows)], ignore_index=True)
            print(f"    Added {len(new_dup_rows)} duplicate records to duplicados_registro.xlsx")

        # Enforce canonical column order (uuid first) to match pipeline output
        _canonical_ann_cols = [
            'uuid', 'specimen_id', 'original_path', 'current_path',
            'macroclass_label', 'class_label', 'genera_label',
            'campaign_year', 'fuente', 'comentarios', 'created_at',
        ]
        def _reorder_columns(df: pd.DataFrame, preferred: list) -> pd.DataFrame:
            ordered = [c for c in preferred if c in df.columns]
            remaining = [c for c in df.columns if c not in ordered]
            return df[ordered + remaining]

        output_ann_copy = _reorder_columns(output_ann_copy, _canonical_ann_cols)
        output_dup_copy = _reorder_columns(output_dup_copy, _canonical_ann_cols)

        if not self.dry_run:
            output_ann_copy.to_excel(output_ann_path, index=False)
            output_hash_copy.to_excel(output_hash_path, index=False)
            if not output_dup_copy.empty:
                output_dup_reg_path.parent.mkdir(parents=True, exist_ok=True)
                output_dup_copy.to_excel(output_dup_reg_path, index=False)

        print(f"    Annotation records processed: {len(s_ann)}")
        print(f"    Duplicates detected: {len(new_dup_rows)}")
        print(f"    New records appended: {len(new_ann_rows)}")
        if rows_auto_merged: print(f"    Fields auto-merged (gap-fill): {rows_auto_merged}")
        if conflicts_resolved: print(f"    Conflicts resolved: {conflicts_resolved}")

    # ------------------------------------------------------------------
    # Merge conflict UI  ([0] existing / [1] staging / [2] custom / [d] defer)
    # ------------------------------------------------------------------

    def _resolve_registry_conflict(
        self,
        registry_name: str,
        match_label: str,
        comparison_table: str,
        discrepancies: Dict[str, Dict[str, str]],
        context_info: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Optional[Dict[str, str]]]:
        """Ask the user to resolve a registry merge conflict via InteractionManager.

        Returns:
            (choice, custom_values)
            choice:
              0  - keep existing (output)
              1  - use staging values
              2  - custom per field (custom_values dict is populated)
              -1 - defer
            custom_values: only set when choice == 2
        """
        decision_request = create_merge_registry_conflict_decision(
            registry_name=registry_name,
            match_label=match_label,
            comparison_table=comparison_table,
            context_info=context_info,
        )
        result = self.interaction_manager.request_decision(decision_request)

        if result.outcome == DecisionOutcome.DEFER:
            return -1, None

        choice = result.selected_option if result.selected_option is not None else 0

        if choice == 2:
            # Custom per field — prompt inline (or auto-keep-existing in auto mode)
            custom_values = {}
            if self.interaction_manager.mode == InteractionMode.AUTO_ACCEPT:
                # Auto mode: keep existing
                return 0, None
            for col, d in discrepancies.items():
                print(f"\n  {col}:  [0] existing = '{d['output']}'  |  [1] staging = '{d['staging']}'")
                val = input(f"  Enter value for '{col}' (or 0/1 to pick): ").strip()
                if val == '0':
                    custom_values[col] = d['output']
                elif val == '1':
                    custom_values[col] = d['staging']
                elif val:
                    custom_values[col] = val
                else:
                    custom_values[col] = d['output']  # default: keep existing
            return 2, custom_values

        if choice == 3:
            # "Defer" option in the options list
            return -1, None

        return choice, None

    # ------------------------------------------------------------------
    # Comparison-table builder (merge conflict UI)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_comparison_table(
        discrepancies: Dict[str, Dict[str, str]],
        match_label: str,
        context_info: Optional[list[str]] = None,
    ) -> str:
        """Return a box-drawing comparison table for a merge conflict.

        Args:
            discrepancies: {col_name: {'staging': val, 'output': val}}
            match_label: human-readable label identifying the matched record
            context_info: Optional list of informational lines to show above
                          the conflict table (e.g. specimen_id, paths).
        """
        field_header = "Field"
        col_a_header = "Existing (output)"
        col_b_header = "Staging"

        field_names = list(discrepancies.keys())

        col_w_field = max(len(field_header), *(len(f) for f in field_names)) if field_names else len(field_header)
        col_w_a = max(
            len(col_a_header),
            *(len(str(d.get('output', '') or '(empty)')) for d in discrepancies.values()),
        ) if discrepancies else len(col_a_header)
        col_w_b = max(
            len(col_b_header),
            *(len(str(d.get('staging', '') or '(empty)')) for d in discrepancies.values()),
        ) if discrepancies else len(col_b_header)

        sep = "─"
        div = [sep * (col_w_field + 2), sep * (col_w_a + 2), sep * (col_w_b + 2)]

        lines: list[str] = []
        if context_info:
            for ci in context_info:
                lines.append(f"  {ci}")
            lines.append("")  # blank separator before the table
        lines.append("┌" + "┬".join(div) + "┐")
        lines.append(
            "│"
            + f" {field_header:<{col_w_field}} │"
            + f" {col_a_header:<{col_w_a}} │"
            + f" {col_b_header:<{col_w_b}} │"
        )
        lines.append("├" + "┼".join(div) + "┤")
        for fname in field_names:
            o_val = str(discrepancies[fname].get('output', '') or '(empty)')
            s_val = str(discrepancies[fname].get('staging', '') or '(empty)')
            # Clean up 'nan' string representation
            o_val = o_val if o_val.lower() != 'nan' else '(empty)'
            s_val = s_val if s_val.lower() != 'nan' else '(empty)'
            lines.append(
                "│"
                + f" {fname:<{col_w_field}} │"
                + f" {o_val:<{col_w_a}} │"
                + f" {s_val:<{col_w_b}} │"
            )
        lines.append("└" + "┴".join(div) + "┘")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core merge logic
    # ------------------------------------------------------------------

    def _merge_dataframes(
        self,
        staging_df: pd.DataFrame,
        output_df: pd.DataFrame,
        match_cols: list[str],
        registry_name: str,
        normalize_fn=None,
    ) -> Optional[pd.DataFrame]:
        """Merge two DataFrames, matching by *match_cols* instead of UUID.

        The first column in *match_cols* that is present in both DataFrames
        is used as the primary match key.  If *normalize_fn* is given it is
        applied to the primary key values before comparison (the original
        values are preserved in the output).

        New records from staging are appended.
        Records with the same key are checked for field discrepancies.

        Args:
            staging_df:   DataFrame from staging
            output_df:    DataFrame from existing output
            match_cols:   Ordered list of candidate match columns
            registry_name: Name of the registry (for logging/UI)
            normalize_fn: Optional callable(str)->str for key normalisation

        Returns:
            Merged DataFrame, or None if no changes
        """

        # --- Pick the first usable match column ---
        key_col: Optional[str] = None
        for col in match_cols:
            if col in staging_df.columns and col in output_df.columns:
                key_col = col
                break

        if key_col is None:
            # No shared match column — just append
            merged = pd.concat([output_df, staging_df], ignore_index=True)
            self.stats['registry_records_merged'] += len(staging_df)
            print(f"    Appended {len(staging_df)} records (no usable match column among {match_cols})")
            return merged

        # --- Helper: make normalised key series ---
        def _make_key(df: pd.DataFrame) -> pd.Series:
            raw = df[key_col].fillna('').astype(str)
            if normalize_fn is not None:
                return raw.map(normalize_fn)
            return raw

        staging_keys = _make_key(staging_df)
        output_keys = _make_key(output_df)

        staging_key_set = set(staging_keys)
        output_key_set = set(output_keys)

        new_keys = staging_key_set - output_key_set - {''}
        overlapping_keys = staging_key_set & output_key_set - {''}

        print(f"    Match column: {key_col}" + (f" (normalised)" if normalize_fn else ""))
        print(f"    Existing records: {len(output_df)}")
        print(f"    Staging records: {len(staging_df)}")
        print(f"    New records: {len(new_keys)}")
        print(f"    Overlapping records: {len(overlapping_keys)}")

        # --- Handle overlapping records ---
        conflicts_resolved = 0
        rows_updated = 0
        skip_cols = {'created_at', 'updated_at', 'timestamp', 'uuid', 'id',
                     'current_path'}  # current_path changes after merge

        for norm_key in overlapping_keys:
            # Find first matching row in each DF
            staging_row = staging_df[staging_keys == norm_key].iloc[0]
            output_idx = output_df[output_keys == norm_key].index[0]
            output_row = output_df.loc[output_idx]

            # Always accumulate original_path
            if 'original_path' in staging_df.columns and 'original_path' in output_df.columns:
                s_orig = str(staging_row.get('original_path', '')) if pd.notna(staging_row.get('original_path')) else ''
                o_orig = str(output_row.get('original_path', '')) if pd.notna(output_row.get('original_path')) else ''
                merged_paths = self._merge_original_paths(o_orig, s_orig)
                if merged_paths != o_orig and not self.dry_run:
                    output_df.at[output_idx, 'original_path'] = merged_paths

            discrepancies: Dict[str, Dict[str, str]] = {}

            for col in staging_df.columns:
                if col in skip_cols or col == key_col or col == 'original_path':
                    continue
                if col not in output_df.columns:
                    continue

                s_val = staging_row.get(col)
                o_val = output_row.get(col)

                s_str = str(s_val) if pd.notna(s_val) else ''
                o_str = str(o_val) if pd.notna(o_val) else ''

                if s_str != o_str:
                    if not o_str and s_str:
                        # Staging fills a gap → auto-merge
                        if not self.dry_run:
                            output_df.at[output_idx, col] = s_val
                        rows_updated += 1
                    elif o_str and not s_str:
                        # Output has data, staging empty → keep output
                        pass
                    else:
                        discrepancies[col] = {
                            'staging': s_str,
                            'output': o_str,
                        }

            if discrepancies:
                match_display = f"{key_col}={staging_row[key_col]}"

                # Build context lines for display and a context dict for the decision payload
                ctx_lines: list[str] = []
                context_dict: Dict[str, str] = {}

                # specimen_id
                if 'specimen_id' in staging_df.columns or 'specimen_id' in output_df.columns:
                    s_spec = str(staging_row.get('specimen_id', '')) if 'specimen_id' in staging_df.columns and pd.notna(staging_row.get('specimen_id', None)) else ''
                    o_spec = str(output_row.get('specimen_id', '')) if 'specimen_id' in output_df.columns and pd.notna(output_row.get('specimen_id', None)) else ''
                    s_spec = s_spec if s_spec and s_spec.lower() != 'nan' else ''
                    o_spec = o_spec if o_spec and o_spec.lower() != 'nan' else ''
                    if o_spec or s_spec:
                        ctx_lines.append(f"specimen_id (existing): {o_spec or '(empty)'}")
                        ctx_lines.append(f"specimen_id (staging):  {s_spec or '(empty)'}")
                    context_dict['specimen_id_existing'] = o_spec or ''
                    context_dict['specimen_id_staging'] = s_spec or ''

                # original_path
                s_orig = str(staging_row.get('original_path', '')) if 'original_path' in staging_df.columns and pd.notna(staging_row.get('original_path', None)) else ''
                o_orig = str(output_row.get('original_path', '')) if 'original_path' in output_df.columns and pd.notna(output_row.get('original_path', None)) else ''
                if o_orig or s_orig:
                    ctx_lines.append(f"original_path (existing): {o_orig or '(empty)'}")
                    ctx_lines.append(f"original_path (staging):  {s_orig or '(empty)'}")
                context_dict['original_path_existing'] = o_orig or ''
                context_dict['original_path_staging'] = s_orig or ''

                # EXIF only when campaign_year differs
                o_year = None
                o_from = None
                s_year = None
                s_from = None
                if 'campaign_year' in discrepancies:
                    o_candidates = [p for p in (o_orig, str(output_row.get('current_path', ''))) if p]
                    s_candidates = [p for p in (s_orig, str(staging_row.get('current_path', ''))) if p]
                    o_year, o_from = self._extract_exif_year_from_paths(o_candidates)
                    s_year, s_from = self._extract_exif_year_from_paths(s_candidates)
                    ctx_lines.append("")
                    ctx_lines.append(f"EXIF year (existing): {o_year}" if o_year else "EXIF year (existing): (not available)")
                    ctx_lines.append(f"EXIF year (staging):  {s_year}" if s_year else "EXIF year (staging):  (not available)")
                    context_dict['exif_existing'] = str(o_year) if o_year else ''
                    context_dict['exif_existing_from'] = str(o_from) if o_from else ''
                    context_dict['exif_staging'] = str(s_year) if s_year else ''
                    context_dict['exif_staging_from'] = str(s_from) if s_from else ''

                # Build table and request decision (include context dict)
                table = self._build_comparison_table(discrepancies, match_display, context_info=ctx_lines)

                # Determine subdir key (handles ;-separated original_path)
                subdir_key = ''
                try:
                    o_orig = context_dict.get('original_path_existing', '')
                    s_orig = context_dict.get('original_path_staging', '')
                    subdir_key = self._extract_subdir_key(o_orig or s_orig)
                except Exception:
                    subdir_key = ''

                # Reuse cached per-subdir choice if present
                cached = self._subdir_merge_choices.get(subdir_key)
                choice = None
                custom_values = None
                if cached is not None:
                    if isinstance(cached, tuple) and cached[0] == 'custom':
                        choice = 2
                        custom_values = cached[1]
                    else:
                        choice = int(cached)
                else:
                    decision_request = create_merge_registry_conflict_decision(
                        registry_name=registry_name,
                        match_label=match_display,
                        comparison_table=table,
                        context_info=context_dict,
                    )
                    result = self.interaction_manager.request_decision(decision_request)

                    if result.outcome == DecisionOutcome.DEFER:
                        choice = -1
                    elif result.outcome == DecisionOutcome.APPLY_TO_SUBDIRECTORY:
                        if result.selected_option is not None:
                            sel = int(result.selected_option)
                            if subdir_key:
                                self._subdir_merge_choices[subdir_key] = sel
                            choice = sel
                        elif result.custom_value:
                            cv = {}
                            for part in str(result.custom_value).split(';'):
                                if '=' in part:
                                    k, v = part.split('=', 1)
                                    cv[k.strip()] = v.strip()
                            if subdir_key:
                                self._subdir_merge_choices[subdir_key] = ('custom', cv)
                            choice = 2
                            custom_values = cv
                        else:
                            choice = 0
                    elif result.outcome == DecisionOutcome.CUSTOM:
                        cv = {}
                        for part in str(result.custom_value).split(';'):
                            if '=' in part:
                                k, v = part.split('=', 1)
                                cv[k.strip()] = v.strip()
                        choice = 2
                        custom_values = cv
                    else:
                        choice = int(result.selected_option) if result.selected_option is not None else 0

                if choice == 1:
                    for col in discrepancies:
                        if not self.dry_run:
                            output_df.at[output_idx, col] = staging_row[col]
                    rows_updated += 1
                elif choice == 2 and custom_values:
                    for col, val in custom_values.items():
                        if not self.dry_run:
                            output_df.at[output_idx, col] = val
                    rows_updated += 1
                elif choice == 0 or choice == -1:
                    pass  # keep existing

                conflicts_resolved += 1
                self.stats['registry_conflicts'] += 1

        # --- Add truly new records ---
        if new_keys:
            new_mask = staging_keys.isin(new_keys)
            new_rows = staging_df[new_mask]
            output_df = pd.concat([output_df, new_rows], ignore_index=True)
            self.stats['registry_records_merged'] += len(new_rows)

        if rows_updated:
            print(f"    Updated {rows_updated} existing records")
        if conflicts_resolved:
            print(f"    Resolved {conflicts_resolved} conflicts")

        return output_df
    
    def _update_registry_paths(self):
        """
        Update file paths in registries to reflect the final output directory.
        
        After merge, the 'current_path' fields in registries still reference
        the staging directory. This updates them to point to the output directory.
        """
        print("\n--- Updating Registry Paths ---")
        
        output_reg_dir = self.output_dir / self.REGISTRY_DIR
        if not output_reg_dir.exists():
            return
        
        staging_str = str(self.staging_dir)
        output_str = str(self.output_dir)
        
        path_columns = ['current_path', 'file_path']
        
        for registry_file in self.REGISTRY_FILES:
            reg_path = output_reg_dir / registry_file
            if not reg_path.exists():
                continue
            
            try:
                df = pd.read_excel(reg_path)
                updated = False
                
                for col in path_columns:
                    if col in df.columns:
                        mask = df[col].astype(str).str.startswith(staging_str)
                        if mask.any():
                            df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(
                                staging_str, output_str, n=1
                            )
                            updated = True
                            count = mask.sum()
                            print(f"  {registry_file}: Updated {count} paths in '{col}'")

                # Apply exact mappings for any duplicates we copied into Duplicados
                if self._copied_duplicates:
                    for col in path_columns:
                        if col in df.columns:
                            for src, dst in self._copied_duplicates.items():
                                exact_mask = df[col].astype(str) == src
                                if exact_mask.any():
                                    df.loc[exact_mask, col] = dst
                                    updated = True
                                    print(f"  {registry_file}: Rewrote {exact_mask.sum()} exact paths in '{col}' to copied duplicates")
                
                if updated and not self.dry_run:
                    df.to_excel(reg_path, index=False)
                    
            except Exception as e:
                logger.error(f"Failed to update paths in {registry_file}: {e}")
                print(f"  ERROR updating {registry_file}: {e}")
    
    def merge(self) -> Dict[str, int]:
        """
        Execute the full merge process.
        
        Returns:
            Dictionary of merge statistics
        """
        print("=" * 60)
        print("MERGE STAGING -> OUTPUT")
        print("=" * 60)
        print(f"Staging:  {self.staging_dir}")
        print(f"Output:   {self.output_dir}")
        print(f"Dry run:  {self.dry_run}")
        print(f"Mode:     {self.interaction_manager.mode.value}")
        
        # Validate
        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory not found: {self.staging_dir}")
        
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 0: Load pre-computed hashes from both registries
        self._load_hash_registries()
        
        # Step 0b: Ensure all staging images have hashes (prompt user if not)
        self._check_missing_hashes()
        # Reset registry-hit counter — _check_missing_hashes also calls
        # _get_file_hashes which increments the stat, but that is just a
        # validation step and should not inflate the final count.
        self.stats['hashes_from_registry'] = 0
        
        # Step 1: Merge files
        self._merge_files()
        
        # Step 2: Merge registries
        self._merge_registries()
        
        # Step 3: Update paths in registries
        self._update_registry_paths()
        
        # Print summary
        print(f"\n{'='*60}")
        print("MERGE COMPLETE")
        print(f"{'='*60}")
        print(f"  Files copied:            {self.stats['files_copied']}")
        print(f"  Duplicates skipped:      {self.stats['files_skipped_duplicate']}")
        if self.stats['files_skipped_near_duplicate']:
            print(f"    (near-duplicates):     {self.stats['files_skipped_near_duplicate']}")
        print(f"  Conflicts (files):       {self.stats['files_skipped_conflict']}")
        print(f"  Registry conflicts:      {self.stats['registry_conflicts']}")
        print(f"  Hashes from registry:    {self.stats['hashes_from_registry']}")
        print(f"  Hashes computed:         {self.stats['hashes_computed']}")
        print(f"  Errors:                  {self.stats['errors']}")
        
        if not self.dry_run and self.stats['errors'] == 0:
            print(f"\n  Staging directory can be safely deleted:")
            print(f"    {self.staging_dir}")
        
        return self.stats


def create_merge_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the merge tool."""
    parser = argparse.ArgumentParser(
        description="Merge a staging directory into the final output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge staging into auto-detected output
  python -m data_ordering.merge_output --staging ./output_MUPA
  
  # Merge with explicit output directory
  python -m data_ordering.merge_output --staging ./output_MUPA --output ./output
  
  # Preview merge without making changes
  python -m data_ordering.merge_output --staging ./output_MUPA --dry-run
  
  # Auto-accept all conflict resolutions
  python -m data_ordering.merge_output --staging ./output_MUPA --auto
        """
    )
    
    parser.add_argument(
        '--staging', '-s',
        type=Path,
        required=True,
        help='Staging directory to merge from'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory to merge into (auto-detected from staging_info.json if omitted)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview merge without making changes'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-resolve all conflicts (skip file duplicates, merge registry gaps)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['interactive', 'deferred', 'auto_accept', 'step_by_step'],
        default=None,
        help='Interaction mode (overrides --auto). Same modes as the main orchestrator.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def main():
    """Main entry point for the merge tool."""
    parser = create_merge_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Validate
    if not args.staging.exists():
        print(f"Error: Staging directory not found: {args.staging}")
        sys.exit(1)
    
    if not args.staging.is_dir():
        print(f"Error: Not a directory: {args.staging}")
        sys.exit(1)
    
    try:
        merger = OutputMerger(
            staging_dir=args.staging.resolve(),
            output_dir=args.output.resolve() if args.output else None,
            dry_run=args.dry_run,
            auto_accept=args.auto,
            interaction_mode=getattr(args, 'mode', None),
        )
        
        stats = merger.merge()
        
        sys.exit(0 if stats['errors'] == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nMerge interrupted!")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
