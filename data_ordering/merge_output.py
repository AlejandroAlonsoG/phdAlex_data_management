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
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import pandas as pd

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
    ):
        """
        Initialize the merger.
        
        Args:
            staging_dir: The staging directory to merge from
            output_dir: The final output directory (auto-detected from staging_info.json if None)
            dry_run: If True, preview changes without modifying anything
            auto_accept: If True, auto-resolve conflicts (skip duplicates, keep existing)
        """
        self.staging_dir = Path(staging_dir)
        self.dry_run = dry_run
        self.auto_accept = auto_accept
        
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
        
        # Stats
        self.stats = {
            'files_copied': 0,
            'files_skipped_duplicate': 0,
            'files_skipped_conflict': 0,
            'directories_created': 0,
            'registry_records_merged': 0,
            'registry_conflicts': 0,
            'errors': 0,
        }
        
        self.conflicts: List[MergeConflict] = []
    
    def _load_staging_info(self) -> Optional[Dict]:
        """Load staging_info.json from the staging directory."""
        info_path = self.staging_dir / "staging_info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()
    
    def _ask_user(self, question: str, options: List[str], default: int = 0) -> int:
        """
        Ask user to choose an option.
        
        Args:
            question: The question to display
            options: List of option descriptions
            default: Default option index
            
        Returns:
            Selected option index
        """
        if self.auto_accept:
            return default
        
        print(f"\n{'─'*50}")
        print(f"CONFLICT: {question}")
        print(f"{'─'*50}")
        for i, opt in enumerate(options):
            marker = " (default)" if i == default else ""
            print(f"  [{i+1}] {opt}{marker}")
        
        while True:
            try:
                choice = input(f"\nSelect option [1-{len(options)}] (Enter for default): ").strip()
                if not choice:
                    return default
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
                print(f"Please enter a number between 1 and {len(options)}")
            except (ValueError, EOFError):
                print(f"Please enter a valid number")
    
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
        """Copy non-registry files from staging to output, handling duplicates."""
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
            output_path = self.output_dir / rel_path
            
            if output_path.exists():
                # File exists in output — check if duplicate
                staging_md5 = self._compute_md5(staging_path)
                output_md5 = self._compute_md5(output_path)
                
                if staging_md5 == output_md5:
                    # Exact duplicate — skip
                    self.stats['files_skipped_duplicate'] += 1
                    logger.debug(f"Skipping exact duplicate: {rel_path}")
                    continue
                else:
                    # Different file at same path — ask user
                    conflict = MergeConflict(
                        conflict_type='file_collision',
                        staging_path=staging_path,
                        output_path=output_path,
                        details={
                            'staging_md5': staging_md5,
                            'output_md5': output_md5,
                            'staging_size': staging_path.stat().st_size,
                            'output_size': output_path.stat().st_size,
                        }
                    )
                    self.conflicts.append(conflict)
                    
                    choice = self._ask_user(
                        f"File collision: {rel_path}\n"
                        f"  Staging: {staging_path.stat().st_size} bytes (MD5: {staging_md5[:12]}...)\n"
                        f"  Output:  {output_path.stat().st_size} bytes (MD5: {output_md5[:12]}...)",
                        [
                            "Keep existing (skip staging file)",
                            "Overwrite with staging file",
                            "Keep both (rename staging file)",
                        ],
                        default=0
                    )
                    
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
                # New file — just copy
                if not self.dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staging_path, output_path)
                self.stats['files_copied'] += 1
            
            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Progress: {i+1}/{total}")
        
        print(f"  Files copied: {self.stats['files_copied']}")
        print(f"  Duplicates skipped: {self.stats['files_skipped_duplicate']}")
        print(f"  Conflicts resolved: {self.stats['files_skipped_conflict']}")
    
    def _merge_registries(self):
        """Merge Excel registry files from staging into output.

        The *anotaciones* and *hashes* registries are merged together in a
        single cross-referenced pass: for each staging image we look for a
        match by hash **or** specimen-id in the output, then reconcile the
        annotation fields.

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
                    staging_df = pd.read_excel(staging_dup_reg)
                    output_df = pd.read_excel(output_dup_reg)
                    merge_result = self._merge_dataframes(
                        staging_df, output_df,
                        match_cols=['specimen_id', 'original_path'],
                        registry_name="duplicados_registro.xlsx",
                        normalize_fn=self._normalize_specimen_id,
                    )
                    if merge_result is not None and not self.dry_run:
                        merge_result.to_excel(output_dup_reg, index=False)
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

    # ------------------------------------------------------------------
    # Unified cross-referenced merge (annotations + hashes)
    # ------------------------------------------------------------------

    def _merge_annotations_and_hashes(
        self,
        staging_reg_dir: Path,
        output_reg_dir: Path,
    ):
        """Merge anotaciones.xlsx and hashes.xlsx together.

        For each staging annotation record the method looks for a match in
        the output by **md5 hash** (primary) or **normalised specimen_id**
        (secondary).  When a match is found every field is compared:

        * Identical / one side fills a gap → auto-merged silently.
        * Real discrepancy → user prompted with [0][1][2][d].

        The hashes registry is kept in sync: overlapping hashes are
        deduplicated and new ones are appended.
        """
        staging_ann_path = staging_reg_dir / "anotaciones.xlsx"
        staging_hash_path = staging_reg_dir / "hashes.xlsx"
        output_ann_path = output_reg_dir / "anotaciones.xlsx"
        output_hash_path = output_reg_dir / "hashes.xlsx"

        # --- Load staging data ---
        s_ann = pd.read_excel(staging_ann_path) if staging_ann_path.exists() else None
        s_hash = pd.read_excel(staging_hash_path) if staging_hash_path.exists() else None

        if s_ann is None and s_hash is None:
            return  # nothing to merge

        # --- If output doesn't exist yet, just copy ---
        ann_is_new = not output_ann_path.exists()
        hash_is_new = not output_hash_path.exists()

        if ann_is_new and s_ann is not None:
            if not self.dry_run:
                shutil.copy2(staging_ann_path, output_ann_path)
            self.stats['registry_records_merged'] += len(s_ann)
            print(f"\n  Merging: anotaciones.xlsx")
            print(f"    Copied {len(s_ann)} records (new registry)")

        if hash_is_new and s_hash is not None:
            if not self.dry_run:
                shutil.copy2(staging_hash_path, output_hash_path)
            self.stats['registry_records_merged'] += len(s_hash)
            print(f"\n  Merging: hashes.xlsx")
            print(f"    Copied {len(s_hash)} records (new registry)")

        if ann_is_new and hash_is_new:
            return  # both were new, nothing to reconcile

        # --- Load output data ---
        o_ann = pd.read_excel(output_ann_path) if output_ann_path.exists() else pd.DataFrame()
        o_hash = pd.read_excel(output_hash_path) if output_hash_path.exists() else pd.DataFrame()

        # Reload staging if they were just copied (we still need to reconcile
        # the other registry that already existed)
        if s_ann is None:
            s_ann = pd.DataFrame()
        if s_hash is None:
            s_hash = pd.DataFrame()

        # If one side was brand-new and just copied, we only need to
        # reconcile the other registry.
        if ann_is_new or hash_is_new:
            # Only the registry that already existed needs merging
            if not ann_is_new and s_ann is not None and len(s_ann):
                print(f"\n  Merging: anotaciones.xlsx")
                result = self._merge_dataframes(
                    s_ann, o_ann,
                    match_cols=['specimen_id', 'original_path'],
                    registry_name='anotaciones.xlsx',
                    normalize_fn=self._normalize_specimen_id,
                )
                if result is not None and not self.dry_run:
                    result.to_excel(output_ann_path, index=False)
            if not hash_is_new and s_hash is not None and len(s_hash):
                print(f"\n  Merging: hashes.xlsx")
                result = self._merge_dataframes(
                    s_hash, o_hash,
                    match_cols=['md5_hash'],
                    registry_name='hashes.xlsx',
                )
                if result is not None and not self.dry_run:
                    result.to_excel(output_hash_path, index=False)
            return

        # ── Both registries exist on both sides → cross-referenced merge ─

        print(f"\n  Cross-referenced merge: anotaciones.xlsx + hashes.xlsx")

        # Ensure all columns that might receive strings are object dtype
        # (pandas may infer float64 for columns that are all-NaN).
        for col in o_ann.columns:
            if o_ann[col].dtype != object:
                o_ann[col] = o_ann[col].astype(object)

        # Build a lookup: staging file_path → md5_hash
        s_path_to_hash: Dict[str, str] = {}
        if 'file_path' in s_hash.columns and 'md5_hash' in s_hash.columns:
            for _, row in s_hash.iterrows():
                fp = str(row['file_path']) if pd.notna(row['file_path']) else ''
                md5 = str(row['md5_hash']) if pd.notna(row['md5_hash']) else ''
                if fp and md5:
                    s_path_to_hash[fp] = md5

        # Build a lookup: output md5_hash → index in o_ann
        # We link o_hash → o_ann via file_path == current_path
        o_hash_to_path: Dict[str, str] = {}
        if 'file_path' in o_hash.columns and 'md5_hash' in o_hash.columns:
            for _, row in o_hash.iterrows():
                fp = str(row['file_path']) if pd.notna(row['file_path']) else ''
                md5 = str(row['md5_hash']) if pd.notna(row['md5_hash']) else ''
                if fp and md5:
                    o_hash_to_path[md5] = fp

        # Build output specimen_id normalised → index in o_ann
        o_specid_norm: Dict[str, int] = {}
        if 'specimen_id' in o_ann.columns:
            for idx, val in o_ann['specimen_id'].items():
                norm = self._normalize_specimen_id(str(val) if pd.notna(val) else '')
                if norm:
                    o_specid_norm[norm] = idx

        # Build output current_path → index in o_ann
        o_curpath_to_idx: Dict[str, int] = {}
        if 'current_path' in o_ann.columns:
            for idx, val in o_ann['current_path'].items():
                cp = str(val) if pd.notna(val) else ''
                if cp:
                    o_curpath_to_idx[cp] = idx

        skip_cols = {'created_at', 'updated_at', 'timestamp', 'uuid', 'id',
                     'current_path'}  # current_path changes after merge
        rows_auto_merged = 0
        conflicts_resolved = 0
        new_count = 0
        processed_output_indices: set = set()

        for s_idx, s_row in s_ann.iterrows():
            # --- Find matching output annotation ---
            matched_idx: Optional[int] = None

            # 1. Match by hash
            s_current = str(s_row.get('current_path', '')) if pd.notna(s_row.get('current_path')) else ''
            s_md5 = s_path_to_hash.get(s_current, '')
            if s_md5 and s_md5 in o_hash_to_path:
                o_file = o_hash_to_path[s_md5]
                if o_file in o_curpath_to_idx:
                    matched_idx = o_curpath_to_idx[o_file]

            # 2. Match by normalised specimen_id
            if matched_idx is None:
                s_specid = str(s_row.get('specimen_id', '')) if pd.notna(s_row.get('specimen_id')) else ''
                s_norm = self._normalize_specimen_id(s_specid)
                if s_norm and s_norm in o_specid_norm:
                    matched_idx = o_specid_norm[s_norm]

            if matched_idx is None:
                # Truly new record
                new_count += 1
                continue

            processed_output_indices.add(matched_idx)
            o_row = o_ann.loc[matched_idx]

            # --- Always accumulate original_path ---
            if 'original_path' in s_ann.columns and 'original_path' in o_ann.columns:
                s_orig = str(s_row.get('original_path', '')) if pd.notna(s_row.get('original_path')) else ''
                o_orig = str(o_row.get('original_path', '')) if pd.notna(o_row.get('original_path')) else ''
                merged_paths = self._merge_original_paths(o_orig, s_orig)
                if merged_paths != o_orig and not self.dry_run:
                    o_ann.at[matched_idx, 'original_path'] = merged_paths

            # --- Field-by-field comparison ---
            discrepancies: Dict[str, Dict[str, str]] = {}

            for col in s_ann.columns:
                if col in skip_cols or col == 'original_path':
                    continue
                if col not in o_ann.columns:
                    continue

                s_val = s_row.get(col)
                o_val = o_row.get(col)

                s_str = str(s_val) if pd.notna(s_val) else ''
                o_str = str(o_val) if pd.notna(o_val) else ''

                if s_str == o_str:
                    continue

                if not o_str and s_str:
                    # Staging fills a gap → auto-merge
                    if not self.dry_run:
                        o_ann.at[matched_idx, col] = s_val
                    rows_auto_merged += 1
                elif o_str and not s_str:
                    # Output already has data → keep it
                    pass
                else:
                    discrepancies[col] = {
                        'staging': s_str,
                        'output': o_str,
                    }

            if not discrepancies:
                continue

            # --- Real conflict → prompt user ---
            s_specid_display = s_row.get('specimen_id', '(no id)')
            match_reason = f"hash={s_md5[:12]}..." if s_md5 else f"specimen_id={s_specid_display}"

            # Build context info for the user
            context_info: list[str] = []
            o_specid_display = o_row.get('specimen_id', '(no id)') if pd.notna(o_row.get('specimen_id', None)) else '(no id)'
            context_info.append(f"specimen_id:  existing={o_specid_display}  |  staging={s_specid_display}")
            o_orig = str(o_row.get('original_path', '')) if pd.notna(o_row.get('original_path', None)) else ''
            s_orig = str(s_row.get('original_path', '')) if pd.notna(s_row.get('original_path', None)) else ''
            if o_orig or s_orig:
                context_info.append(f"original_path (existing): {o_orig or '(empty)'}")
                context_info.append(f"original_path (staging):  {s_orig or '(empty)'}")
            o_cur = str(o_row.get('current_path', '')) if pd.notna(o_row.get('current_path', None)) else ''
            s_cur = str(s_row.get('current_path', '')) if pd.notna(s_row.get('current_path', None)) else ''
            if o_cur or s_cur:
                context_info.append(f"current_path (existing):  {o_cur or '(empty)'}")
                context_info.append(f"current_path (staging):   {s_cur or '(empty)'}")

            table = self._build_comparison_table(discrepancies, match_reason, context_info=context_info)

            choice = self._ask_merge_conflict(
                header=f"Merge conflict ({match_reason})",
                table=table,
            )

            if choice == 0:
                # Keep existing (output) values
                pass
            elif choice == 1:
                # Use staging values
                for col in discrepancies:
                    if not self.dry_run:
                        o_ann.at[matched_idx, col] = s_row[col]
            elif choice == 2:
                # Custom per field
                for col, d in discrepancies.items():
                    custom_val = self._ask_custom_field(col, d['output'], d['staging'])
                    if custom_val is not None and not self.dry_run:
                        o_ann.at[matched_idx, col] = custom_val
            elif choice == -1:
                # Defer — keep output as-is for now
                pass

            conflicts_resolved += 1
            self.stats['registry_conflicts'] += 1

        # Append truly new annotation rows
        if new_count:
            new_indices = []
            for s_idx2, s_row2 in s_ann.iterrows():
                m_idx = None
                sc = str(s_row2.get('current_path', '')) if pd.notna(s_row2.get('current_path')) else ''
                smd5 = s_path_to_hash.get(sc, '')
                if smd5 and smd5 in o_hash_to_path:
                    of = o_hash_to_path[smd5]
                    if of in o_curpath_to_idx:
                        m_idx = o_curpath_to_idx[of]
                if m_idx is None:
                    ssp = str(s_row2.get('specimen_id', '')) if pd.notna(s_row2.get('specimen_id')) else ''
                    sn = self._normalize_specimen_id(ssp)
                    if sn and sn in o_specid_norm:
                        m_idx = o_specid_norm[sn]
                if m_idx is None:
                    new_indices.append(s_idx2)

            if new_indices:
                new_rows = s_ann.loc[new_indices]
                o_ann = pd.concat([o_ann, new_rows], ignore_index=True)
                self.stats['registry_records_merged'] += len(new_rows)

        if not self.dry_run:
            o_ann.to_excel(output_ann_path, index=False)

        # --- Merge hashes registry (simple dedup by md5_hash) ---
        if len(s_hash) and len(o_hash):
            existing_hashes = set(o_hash['md5_hash'].dropna().astype(str))
            new_hash_rows = s_hash[~s_hash['md5_hash'].astype(str).isin(existing_hashes)]
            if len(new_hash_rows):
                o_hash = pd.concat([o_hash, new_hash_rows], ignore_index=True)
                self.stats['registry_records_merged'] += len(new_hash_rows)
            if not self.dry_run:
                o_hash.to_excel(output_hash_path, index=False)

        print(f"    Existing annotation records: {len(o_ann) - new_count}")
        print(f"    Staging annotation records: {len(s_ann)}")
        print(f"    Matched (by hash or ID): {len(s_ann) - new_count}")
        print(f"    New records appended: {new_count}")
        if rows_auto_merged:
            print(f"    Fields auto-merged (gap-fill): {rows_auto_merged}")
        if conflicts_resolved:
            print(f"    Conflicts resolved: {conflicts_resolved}")

    # ------------------------------------------------------------------
    # Merge conflict UI  ([0] existing / [1] staging / [2] custom / [d] defer)
    # ------------------------------------------------------------------

    def _ask_merge_conflict(self, header: str, table: str) -> int:
        """Prompt the user about a merge conflict.

        Returns:
            0  – keep existing (output)
            1  – use staging values
            2  – enter custom value per field
            -1 – defer
        """
        if self.auto_accept:
            return 0  # keep existing is the safe default

        print(f"\n{'─'*60}")
        print(f"CONFLICT: {header}")
        print(f"{'─'*60}")
        print(table)
        print()
        print("  [0] Keep existing (output) values")
        print("  [1] Use staging values")
        print("  [2] Enter custom value for each field")
        print("  [d] Defer (leave for later)")

        while True:
            try:
                choice = input("\nEnter choice (0, 1, 2, d): ").strip().lower()
                if choice == '0':
                    return 0
                elif choice == '1':
                    return 1
                elif choice == '2':
                    return 2
                elif choice == 'd':
                    return -1
                else:
                    print("Invalid input. Enter 0, 1, 2, or d.")
            except (KeyboardInterrupt, EOFError):
                print("\nDeferred.")
                return -1

    def _ask_custom_field(
        self, field_name: str, output_val: str, staging_val: str,
    ) -> Optional[str]:
        """Ask the user for a custom value for a single field."""
        if self.auto_accept:
            return None  # keep existing
        print(f"\n  {field_name}:  [0] existing = '{output_val}'  |  [1] staging = '{staging_val}'")
        val = input(f"  Enter value for '{field_name}' (or 0/1 to pick): ").strip()
        if val == '0':
            return output_val
        elif val == '1':
            return staging_val
        elif val:
            return val
        return None  # keep existing

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

        col_w_field = max(len(field_header), *(len(f) for f in field_names))
        col_w_a = max(
            len(col_a_header),
            *(len(d['output']) for d in discrepancies.values()),
        )
        col_w_b = max(
            len(col_b_header),
            *(len(d['staging']) for d in discrepancies.values()),
        )

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
            o_val = discrepancies[fname]['output']
            s_val = discrepancies[fname]['staging']
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

                # Build context info for the user
                ctx_info: list[str] = []
                for info_col in ['specimen_id', 'original_path', 'current_path']:
                    if info_col in staging_df.columns or info_col in output_df.columns:
                        s_v = str(staging_row.get(info_col, '')) if info_col in staging_df.columns and pd.notna(staging_row.get(info_col, None)) else ''
                        o_v = str(output_row.get(info_col, '')) if info_col in output_df.columns and pd.notna(output_row.get(info_col, None)) else ''
                        if s_v or o_v:
                            ctx_info.append(f"{info_col} (existing): {o_v or '(empty)'}")
                            ctx_info.append(f"{info_col} (staging):  {s_v or '(empty)'}")

                table = self._build_comparison_table(discrepancies, match_display, context_info=ctx_info)

                choice = self._ask_user(
                    f"Registry conflict in {registry_name} ({match_display}):\n{table}",
                    [
                        "Keep existing values",
                        "Use staging values",
                        "Merge (staging fills gaps, keep existing where both have values)",
                    ],
                    default=2,
                )

                if choice == 1:
                    for col in discrepancies:
                        if not self.dry_run:
                            output_df.at[output_idx, col] = staging_row[col]
                    rows_updated += 1
                elif choice == 2:
                    for col, d in discrepancies.items():
                        if not d['output']:
                            if not self.dry_run:
                                output_df.at[output_idx, col] = staging_row[col]
                    rows_updated += 1
                # choice == 0: keep existing

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
        print(f"Auto:     {self.auto_accept}")
        
        # Validate
        if not self.staging_dir.exists():
            raise FileNotFoundError(f"Staging directory not found: {self.staging_dir}")
        
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        print(f"  Conflicts (files):       {self.stats['files_skipped_conflict']}")
        print(f"  Registry records merged: {self.stats['registry_records_merged']}")
        print(f"  Registry conflicts:      {self.stats['registry_conflicts']}")
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
