"""Remove duplicate UUIDs from the hash registry.

Usage:
    python -m data_ordering.remove_duplicates_from_excel --output-dir <orchestrator_output_dir> [--dry-run] [--backup]

This script looks for <output_dir>/Duplicados/duplicados_registro.xlsx and
<output_dir>/registries/hashes.xlsx and removes any rows in the latter whose
`uuid` is present in the duplicates registry.
"""
from pathlib import Path
import argparse
import shutil
import sys
import datetime
import pandas as pd


def find_uuid_column(df: pd.DataFrame):
    # Prefer exact 'uuid' case-insensitive, then any column containing 'uuid'
    for col in df.columns:
        if col.lower() == 'uuid':
            return col
    for col in df.columns:
        if 'uuid' in col.lower():
            return col
    return None


def main():
    p = argparse.ArgumentParser(description="Remove duplicate UUIDs from hash registry")
    p.add_argument('--output-dir', '-o', required=True, help='Orchestrator output directory (staging or final)')
    p.add_argument('--dry-run', action='store_true', help='Do not modify files, only report')
    p.add_argument('--backup', action='store_true', help='Create a timestamped backup of hashes.xlsx before modifying')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    dup_path = out_dir / 'Duplicados' / 'duplicados_registro.xlsx'
    hash_path = out_dir / 'registries' / 'hashes.xlsx'

    if not dup_path.exists():
        print(f"No duplicates registry found at: {dup_path}")
        sys.exit(1)
    if not hash_path.exists():
        print(f"No hash registry found at: {hash_path}")
        sys.exit(1)

    dup_df = pd.read_excel(dup_path)
    uuid_col_dup = find_uuid_column(dup_df)
    if not uuid_col_dup:
        print("Could not find a UUID column in duplicates registry")
        sys.exit(1)

    dup_uuids = set(v for v in dup_df[uuid_col_dup].astype(str).dropna().str.strip())
    if not dup_uuids:
        print("No UUIDs found in duplicates registry; nothing to do.")
        return

    hash_df = pd.read_excel(hash_path)
    uuid_col_hash = find_uuid_column(hash_df)
    if not uuid_col_hash:
        print("Could not find a UUID column in hash registry")
        sys.exit(1)

    initial_count = len(hash_df)
    mask_remove = hash_df[uuid_col_hash].astype(str).isin(dup_uuids)
    remove_count = int(mask_remove.sum())

    if remove_count == 0:
        print(f"No matching UUIDs found in {hash_path}. Nothing removed.")
        return

    print(f"Found {remove_count} matching hash entries out of {initial_count} rows.\n")
    if args.dry_run:
        print("Dry run: not modifying files.")
        return

    # Backup if requested
    if args.backup:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = hash_path.with_suffix(f".xlsx.bak_{ts}")
        shutil.copy2(hash_path, backup_path)
        print(f"Backup created: {backup_path}")

    # Remove rows and save
    new_df = hash_df[~mask_remove].reset_index(drop=True)
    new_df.to_excel(hash_path, index=False)
    print(f"Removed {remove_count} rows from {hash_path}. New row count: {len(new_df)}")


if __name__ == '__main__':
    main()
