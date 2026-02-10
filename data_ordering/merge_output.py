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
        """Merge Excel registry files from staging into output."""
        print("\n--- Merging Registries ---")
        
        staging_reg_dir = self.staging_dir / self.REGISTRY_DIR
        output_reg_dir = self.output_dir / self.REGISTRY_DIR
        
        if not staging_reg_dir.exists():
            print("  No registries in staging directory")
            return
        
        if not self.dry_run:
            output_reg_dir.mkdir(parents=True, exist_ok=True)
        
        for registry_file in self.REGISTRY_FILES:
            staging_reg = staging_reg_dir / registry_file
            output_reg = output_reg_dir / registry_file
            
            if not staging_reg.exists():
                continue
            
            print(f"\n  Merging: {registry_file}")
            
            if not output_reg.exists():
                # No existing registry — just copy
                if not self.dry_run:
                    shutil.copy2(staging_reg, output_reg)
                staging_df = pd.read_excel(staging_reg)
                self.stats['registry_records_merged'] += len(staging_df)
                print(f"    Copied {len(staging_df)} records (new registry)")
                continue
            
            # Both exist — merge
            try:
                staging_df = pd.read_excel(staging_reg)
                output_df = pd.read_excel(output_reg)
                
                # Determine the key column for dedup
                if registry_file == "anotaciones.xlsx":
                    key_col = 'uuid'
                    merge_result = self._merge_dataframes(
                        staging_df, output_df, key_col, registry_file
                    )
                elif registry_file == "hashes.xlsx":
                    key_col = 'uuid'
                    merge_result = self._merge_dataframes(
                        staging_df, output_df, key_col, registry_file
                    )
                else:
                    key_col = 'id'
                    merge_result = self._merge_dataframes(
                        staging_df, output_df, key_col, registry_file
                    )
                
                if merge_result is not None and not self.dry_run:
                    merge_result.to_excel(output_reg, index=False)
                
            except Exception as e:
                logger.error(f"Failed to merge {registry_file}: {e}")
                self.stats['errors'] += 1
                print(f"    ERROR: {e}")
        
        # Handle duplicate registry (lives in Duplicados/ not registries/)
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
                        staging_df, output_df, 'uuid', "duplicados_registro.xlsx"
                    )
                    if merge_result is not None and not self.dry_run:
                        merge_result.to_excel(output_dup_reg, index=False)
            except Exception as e:
                logger.error(f"Failed to merge duplicate registry: {e}")
                self.stats['errors'] += 1
                print(f"    ERROR: {e}")
    
    def _merge_dataframes(
        self,
        staging_df: pd.DataFrame,
        output_df: pd.DataFrame,
        key_col: str,
        registry_name: str,
    ) -> Optional[pd.DataFrame]:
        """
        Merge two DataFrames, handling conflicts.
        
        New records from staging are appended.
        Records with same key are checked for discrepancies.
        
        Args:
            staging_df: DataFrame from staging
            output_df: DataFrame from existing output
            key_col: Column to use as unique key
            registry_name: Name of the registry for user messages
            
        Returns:
            Merged DataFrame or None if no changes
        """
        if key_col not in staging_df.columns or key_col not in output_df.columns:
            # Can't merge by key — just append
            merged = pd.concat([output_df, staging_df], ignore_index=True)
            self.stats['registry_records_merged'] += len(staging_df)
            print(f"    Appended {len(staging_df)} records (no key column '{key_col}')")
            return merged
        
        # Find new vs existing records
        existing_keys = set(output_df[key_col].dropna().astype(str))
        staging_keys = set(staging_df[key_col].dropna().astype(str))
        
        new_keys = staging_keys - existing_keys
        overlapping_keys = staging_keys & existing_keys
        
        print(f"    Existing records: {len(output_df)}")
        print(f"    Staging records: {len(staging_df)}")
        print(f"    New records: {len(new_keys)}")
        print(f"    Overlapping records: {len(overlapping_keys)}")
        
        # Handle overlapping records — check for discrepancies
        conflicts_resolved = 0
        rows_updated = 0
        
        for key in overlapping_keys:
            staging_row = staging_df[staging_df[key_col].astype(str) == key].iloc[0]
            output_idx = output_df[output_df[key_col].astype(str) == key].index[0]
            output_row = output_df.loc[output_idx]
            
            # Find columns with discrepancies (excluding timestamp columns)
            skip_cols = {'created_at', 'updated_at', 'timestamp'}
            discrepancies = {}
            
            for col in staging_df.columns:
                if col in skip_cols or col == key_col:
                    continue
                if col not in output_df.columns:
                    continue
                    
                staging_val = staging_row.get(col)
                output_val = output_row.get(col)
                
                # Normalize for comparison
                s_str = str(staging_val) if pd.notna(staging_val) else ''
                o_str = str(output_val) if pd.notna(output_val) else ''
                
                if s_str != o_str:
                    # Check if staging fills in a gap (output is empty)
                    if not o_str and s_str:
                        # Staging has data, output doesn't — auto-merge
                        if not self.dry_run:
                            output_df.at[output_idx, col] = staging_val
                        rows_updated += 1
                    elif o_str and not s_str:
                        # Output has data, staging doesn't — keep output
                        pass
                    else:
                        # Both have different data — this is a real conflict
                        discrepancies[col] = {
                            'staging': s_str,
                            'output': o_str,
                        }
            
            if discrepancies:
                # Ask user about discrepancies
                disc_summary = "\n".join(
                    f"    {col}: staging='{d['staging']}' vs output='{d['output']}'"
                    for col, d in discrepancies.items()
                )
                
                choice = self._ask_user(
                    f"Registry conflict for {key_col}={key} in {registry_name}:\n{disc_summary}",
                    [
                        "Keep existing values",
                        "Use staging values",
                        "Merge (staging fills gaps, keep existing where both have values)",
                    ],
                    default=2  # Merge is safest
                )
                
                if choice == 1:
                    # Use staging values for conflicting fields
                    for col in discrepancies:
                        if not self.dry_run:
                            output_df.at[output_idx, col] = staging_row[col]
                    rows_updated += 1
                elif choice == 2:
                    # Merge: staging fills gaps only
                    for col, d in discrepancies.items():
                        if not d['output']:
                            if not self.dry_run:
                                output_df.at[output_idx, col] = staging_row[col]
                    rows_updated += 1
                # choice == 0: keep existing (do nothing)
                
                conflicts_resolved += 1
                self.stats['registry_conflicts'] += 1
        
        # Add new records
        if new_keys:
            new_rows = staging_df[staging_df[key_col].astype(str).isin(new_keys)]
            output_df = pd.concat([output_df, new_rows], ignore_index=True)
            self.stats['registry_records_merged'] += len(new_keys)
        
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
        print("MERGE STAGING → OUTPUT")
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
