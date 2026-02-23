"""Minimal merger helper utilities (work-in-progress).

This file implements a small, focused helper used as the first step of
the manual merger workflow: extract a mapping of UUID -> (pHash, MD5)
from a repository's registries (assumes `registries/hashes.xlsx`).

The implementation intentionally mirrors the registry-loading
behaviour in `merge_output.py` (use of `hashes.xlsx`, tolerant
column name handling) but keeps the function small and non-mutating.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import random


# Function to extract a mapping of uuid-(phash, hash) from one main directory (assuming the excel in under registries)
def extract_uuid_hash_mapping(main_dir: Path) -> Dict[str, Tuple[str, str]]:
    """Extract mapping uuid -> (phash, md5_hash) from a main directory.

    Args:
        main_dir: path to the main directory that should contain
            a `registries/hashes.xlsx` file.

    Returns:
        Dict mapping UUID (string) -> (phash, md5_hash) where missing
        values are empty strings. If the registry file is missing or an
        error occurs an empty dict is returned.
    """
    main_dir = Path(main_dir)
    reg_path = main_dir / "registries" / "hashes.xlsx"
    if not reg_path.exists():
        print(f"No hashes.xlsx found at: {reg_path}")
        return {}

    try:
        df = pd.read_excel(reg_path)
    except Exception as e:
        print(f"Failed to read {reg_path}: {e}")
        return {}

    # tolerant column detection (case-insensitive alternatives)
    cols = {c.lower(): c for c in df.columns}

    uuid_col = cols.get('uuid') or cols.get('id') or None
    md5_col = cols.get('md5_hash') or cols.get('md5') or cols.get('md5hash')
    phash_col = cols.get('phash') or cols.get('p_hash') or cols.get('phash_value')

    mapping: Dict[str, Tuple[str, str]] = {}

    if uuid_col is None:
        print(f"hashes.xlsx at {reg_path} has no 'uuid' column")
        return {}

    for _, row in df.iterrows():
        try:
            raw_uuid = row.get(uuid_col)
        except Exception:
            raw_uuid = None

        if pd.isna(raw_uuid) or raw_uuid is None:
            continue

        uid = str(raw_uuid).strip()
        # md5/phash may be missing  normalise to empty string
        phash = ''
        md5 = ''
        try:
            if phash_col and pd.notna(row.get(phash_col)):
                phash = str(row.get(phash_col))
        except Exception:
            phash = ''
        try:
            if md5_col and pd.notna(row.get(md5_col)):
                md5 = str(row.get(md5_col))
        except Exception:
            md5 = ''

        mapping[uid] = (phash, md5)

    print(f"Loaded {len(mapping)} hash entries from {reg_path}")
    return mapping

# Function to compare all vs all the mappings
## If phash is not present, hash is used. Otherwise, phash is used with a threshold such as the one on main orchestator
## Returns a list with the uuids that are not duplicated, and another one with the ones that are as tuples


def _hex_hamming_distance(h1: str, h2: str) -> Optional[int]:
    """Compute hamming distance between two hex-encoded hashes.

    Returns None if conversion fails or lengths differ.
    """
    try:
        # Normalize: remove 0x if present and whitespace
        a = h1.strip().lower().lstrip('0x')
        b = h2.strip().lower().lstrip('0x')
        if len(a) != len(b):
            return None
        ai = int(a, 16)
        bi = int(b, 16)
        x = ai ^ bi
        # bit_count available in Python 3.8+
        return x.bit_count()
    except Exception:
        return None


def compare_mappings(
    a: Dict[str, Tuple[str, str]],
    b: Dict[str, Tuple[str, str]],
    phash_threshold: int = 8,
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]], List[Tuple[str, str, str, str]]]:
    """Compare two uuid->(phash,md5) mappings.

    Matching priority: perceptual hash (pHash) exact or near (hamming <= threshold),
    then md5 exact.

    Returns (unique_in_a, unique_in_b, duplicates) where
      - unique_in_a/b are dicts of entries present only in that mapping
      - duplicates is a list of tuples: (uuid_a, uuid_b, match_type, match_value)
        where match_type is 'phash_near'|'phash_exact'|'md5'
    """
    # Build reverse indexes for b for quick lookup
    phash_to_uuids_b = {}
    md5_to_uuids_b = {}
    for uid, (phash, md5) in b.items():
        if phash:
            phash_to_uuids_b.setdefault(phash, []).append(uid)
        if md5:
            md5_to_uuids_b.setdefault(md5, []).append(uid)

    matched_b = set()
    duplicates = []

    for ua, (aphash, amd5) in a.items():
        found = False
        # 1) exact phash
        if aphash and aphash in phash_to_uuids_b:
            ub = phash_to_uuids_b[aphash][0]
            duplicates.append((ua, ub, 'phash_exact', aphash))
            matched_b.add(ub)
            found = True
            continue

        # 2) near phash (hamming)
        if aphash:
            for bphash, candidates in phash_to_uuids_b.items():
                hd = _hex_hamming_distance(aphash, bphash)
                if hd is not None and 0 < hd <= phash_threshold:
                    ub = candidates[0]
                    duplicates.append((ua, ub, 'phash_near', f"{aphash}~{bphash} (hd={hd})"))
                    matched_b.add(ub)
                    found = True
                    break
            if found:
                continue

        # 3) md5 exact
        if amd5 and amd5 in md5_to_uuids_b:
            ub = md5_to_uuids_b[amd5][0]
            duplicates.append((ua, ub, 'md5', amd5))
            matched_b.add(ub)
            found = True
            continue

    # Build uniques
    unique_in_a = {k: v for k, v in a.items() if all(dup[0] != k for dup in duplicates)}
    unique_in_b = {k: v for k, v in b.items() if k not in matched_b}

    return unique_in_a, unique_in_b, duplicates

# Function to obtain the extracted data of a uuid from the registry
## It assumes the annotation file exists, opens it and return the extracted information
## It can receive a list of uuids for efficiency


def _find_uuid_column(df):
    # Case-insensitive search for a uuid-like column name
    for c in df.columns:
        if str(c).lower() == 'uuid':
            return c
    for c in df.columns:
        if 'uuid' in str(c).lower() or 'id' == str(c).lower():
            return c
    return None


def get_records_by_uuid(main_dir: Path, uuids: List[str]) -> Dict[str, dict]:
    """Return a mapping uuid -> record-dict from `registries/anotaciones.xlsx`.

    Args:
        main_dir: base directory containing `registries/anotaciones.xlsx`.
        uuids: list of UUID strings to fetch.

    Returns:
        Dict mapping uuid -> row-as-dict for found records (missing UUIDs omitted).
    """
    main_dir = Path(main_dir)
    reg_path = main_dir / 'registries' / 'anotaciones.xlsx'
    if not reg_path.exists():
        print(f"No anotaciones.xlsx found at: {reg_path}")
        return {}

    try:
        df = pd.read_excel(reg_path)
    except Exception as e:
        print(f"Failed to read {reg_path}: {e}")
        return {}

    uuid_col = _find_uuid_column(df)
    if uuid_col is None:
        print(f"Could not find a UUID column in {reg_path}")
        return {}

    # Normalize UUIDs for comparison
    want = set(str(u).strip() for u in uuids if u)
    results: Dict[str, dict] = {}
    for _, row in df.iterrows():
        try:
            raw = row.get(uuid_col)
        except Exception:
            raw = None
        if pd.isna(raw) or raw is None:
            continue
        uid = str(raw).strip()
        if uid in want:
            # Convert row to dict and ensure simple types
            rec = {str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            results[uid] = rec

    return results


# Function to merge the fields of two duplicated records
## It recives a tuple of uuids
## Then it extract their fields
## Then it tries to merge them
### If there is no discrepancy, perfect (NonevsNone, NonevsValue or ValuevsValue with same value)
### If there is a discrepancy, it treats it as in the main_orchestator (refer to original code)


def _normalize_val(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def merge_record_pair(o_record: dict, s_record: dict, auto_choice: str = None):
    """Merge two annotation records (existing, staging) field-by-field.

    Rules:
      - None vs None -> None
      - None vs value -> value
      - value vs value equal -> value
      - value vs value different -> prompt user for choice

    Parameters:
      o_record: dict for existing/output record
      s_record: dict for staging record
      auto_choice: if provided, one of '0','1','2','d' to auto-select choice

    Returns:
      (choice, merged_record) where choice is 0 (keep existing), 1 (use staging),
      2 (custom applied), -1 (deferred). merged_record is the resulting dict
      (or None if deferred).
    """
    # Collect all columns
    cols = list(set(list(o_record.keys()) + list(s_record.keys())))
    # Ensure uuid present and same handling
    merged = {}
    discrepancies = {}

    for col in cols:
        o_val = _normalize_val(o_record.get(col))
        s_val = _normalize_val(s_record.get(col))

        if o_val is None and s_val is None:
            merged[col] = None
        elif o_val is None and s_val is not None:
            merged[col] = s_val
        elif o_val is not None and s_val is None:
            merged[col] = o_val
        else:
            # both have value
            if str(o_val) == str(s_val):
                merged[col] = o_val
            else:
                discrepancies[col] = {'output': o_val, 'staging': s_val}

    if not discrepancies:
        return 0, merged

    # Prepare context for user prompt similar to merge_output
    sample_ctx = []
    try:
        sample_ctx.append(f"specimen_id (existing): {o_record.get('specimen_id', '(no id)')}")
        sample_ctx.append(f"specimen_id (staging):  {s_record.get('specimen_id', '(no id)')}")
    except Exception:
        pass

    # If auto_choice provided, honor it
    if auto_choice is not None:
        choice = auto_choice
    else:
        # Prompt user with options: [0] existing / [1] staging / [2] custom / [d] defer / [a] apply to subdir
        print("\nMerge conflict detected for record pair:")
        for line in sample_ctx:
            print("  ", line)
        print("\nDiscrepant fields:")
        for col, d in discrepancies.items():
            print(f"  {col}: existing='{d['output']}' | staging='{d['staging']}'")
        print("\nOptions: [0] keep existing | [1] use staging | [2] custom per-field | [d] defer | [a] apply to subdirectory")
        choice = input("Select option (0/1/2/d/a): ").strip().lower()

    if choice == 'd':
        return -1, None

    if choice == 'a':
        # Ask which numbered option to apply to the whole subdirectory
        sub_choice = input(f"Which option to apply to the whole subdirectory? [0/1/2]: ").strip()
        if sub_choice.isdigit() and int(sub_choice) in (0, 1, 2):
            sel = int(sub_choice)
            print(f"Applying option {sel} to subdirectory (this selection will be used for all files in subdir)")
            choice = str(sel)
        else:
            print("Invalid sub-option. Defaulting to keep existing (0)")
            choice = '0'

    if choice == '0':
        # keep existing for all discrepant fields
        for col in discrepancies:
            merged[col] = discrepancies[col]['output']
        return 0, merged

    if choice == '1':
        # use staging values
        for col in discrepancies:
            merged[col] = discrepancies[col]['staging']
        return 1, merged

    if choice == '2':
        custom = {}
        if auto_choice is not None:
            # auto mode: default to keep existing
            for col in discrepancies:
                custom[col] = discrepancies[col]['output']
        else:
            for col, d in discrepancies.items():
                print(f"\nField: {col}")
                print(f"  [0] existing = '{d['output']}'")
                print(f"  [1] staging  = '{d['staging']}'")
                val = input(f"  Enter value for '{col}' (or 0/1 to pick): ").strip()
                if val == '0':
                    custom[col] = d['output']
                elif val == '1':
                    custom[col] = d['staging']
                elif val:
                    custom[col] = val
                else:
                    custom[col] = d['output']
        for col, v in custom.items():
            merged[col] = v
        return 2, merged

    # Fallback: keep existing
    for col in discrepancies:
        merged[col] = discrepancies[col]['output']
    return 0, merged




# Main function
# 1. it extracts the mapping of uuid-(phash, hash) from the source directories
# 2. it compares the mappings to find unique and duplicated uuids
# 3. It generates the merged directories
# 4. On the unique uuids, it basicically copies them to their respecitve places (same subtrees as they were but on the new directory). It copies their entries of annotations and hashes
# 5. On the duplicated uuids, it merge their fields (maybe asking the user in between), and then copies the origiinal with the others, updating annotation and hashes accordingly. The duplicated one goes sent to duplicates, updating its own registry and also its hash function


# Testing
if __name__ == '__main__':
    # Quick test entrypoint as requested by the user
    test_dir = Path(r"C:\merge_3a")
    mapping = extract_uuid_hash_mapping(test_dir)
    print(f"Original mapping size: {len(mapping)}")

    if not mapping:
        print("No entries to test; exiting.")
    else:
        # Build modified mapping_b from mapping (shallow copy then edits)
        mapping_b = {}
        i=0
        for k, v in mapping.items():
            mapping_b["REPLACED_" + k] = v
            i+=1
            if i >= 10:
                break

        first_keys = list(mapping.keys())[:10]

        # 2) pick 3 of the remaining first_keys and tweak last char of hashes
        remaining = [k for k in first_keys if k in mapping_b]
        tweak_count = min(3, len(remaining))
        tweaks = random.sample(remaining, tweak_count) if tweak_count else []

        def _tweak_hex_last_char(s: str) -> str:
            if not s:
                return s
            s = str(s)
            last = s[-1]
            try:
                val = int(last, 16)
                val = (val + 1) % 16
                new_last = format(val, 'x')
                return s[:-1] + new_last
            except Exception:
                # fallback: toggle last char between '0' and '1'
                return s[:-1] + ('1' if last != '1' else '0')

        for k in tweaks:
            ph, md = mapping_b.get(k, ('', ''))
            mapping_b[k] = (_tweak_hex_last_char(ph), _tweak_hex_last_char(md))

        # 3) add 5 random new entries describing the change
        for i in range(1, 6):
            uid = f"NEW_added_test_{i}"
            # generate random 16-hex char values
            ph = format(random.getrandbits(64), 'x')
            md = format(random.getrandbits(128), 'x')
            mapping_b[uid] = (ph, md)

        print(f"Modified mapping_b size: {len(mapping_b)}")

        # Compare
        unique_a, unique_b, duplicates = compare_mappings(mapping, mapping_b, phash_threshold=8)

        print(f"Duplicates found: {len(duplicates)}")
        for d in duplicates[:20]:
            print("  ", d)
        print(f"Unique in original (sample 10): {list(unique_a.keys())[:10]}")
        print(f"Unique in modified (sample 10): {list(unique_b.keys())[:10]}")

        # --- New: fetch 3 UUID records from anotaciones.xlsx and print them ---
        # Pick up to 3 UUIDs from the mapping
        sample_uuids = list(mapping.keys())[:3]
        print(f"\nFetching {len(sample_uuids)} records from anotaciones.xlsx for UUIDs: {sample_uuids}")
        recs = get_records_by_uuid(test_dir, sample_uuids)
        if not recs:
            print("No annotation records found for those UUIDs (or anotaciones.xlsx missing).")
        else:
            for uid in sample_uuids:
                r = recs.get(uid)
                print(f"\nUUID: {uid}")
                if not r:
                    print("  Not found in anotaciones.xlsx")
                else:
                    # Print a compact view
                    for k, v in r.items():
                        print(f"  {k}: {v}")

            # --- Now test merge_record_pair with fabricated scenarios ---
            print("\n--- Testing merge_record_pair scenarios (auto-choice=1 to use staging) ---")

            # Build simple demo pairs
            # Base row template: use first found record as template
            template = recs.get(sample_uuids[0]) or {}

            def copy_row_with_changes(base, changes: dict, uuid: str):
                r = dict(base) if base else {}
                r.update(changes)
                r['uuid'] = uuid
                return r

            # 1) none vs none -> both missing for field 'genera_label'
            o1 = copy_row_with_changes(template, {'genera_label': None}, 'MERGE_NONE_NONE_O')
            s1 = copy_row_with_changes(template, {'genera_label': None}, 'MERGE_NONE_NONE_S')

            # 2) none vs value -> staging has value
            o2 = copy_row_with_changes(template, {'campaign_year': None}, 'MERGE_NONE_VAL_O')
            s2 = copy_row_with_changes(template, {'campaign_year': 1999}, 'MERGE_NONE_VAL_S')

            # 3) value vs value equal
            o3 = copy_row_with_changes(template, {'fuente': 'LH'}, 'MERGE_VAL_EQ_O')
            s3 = copy_row_with_changes(template, {'fuente': 'LH'}, 'MERGE_VAL_EQ_S')

            # 4) value vs value different (will prompt if interactive)
            o4 = copy_row_with_changes(template, {'comentarios': 'From output'}, 'MERGE_VAL_DIFF_O')
            s4 = copy_row_with_changes(template, {'comentarios': 'From staging'}, 'MERGE_VAL_DIFF_S')

            cases = [ (o1,s1,'none-none'), (o2,s2,'none-value'), (o3,s3,'value-equal'), (o4,s4,'value-diff') ]
            import sys
            for idx, (o_row, s_row, label) in enumerate(cases, start=1):
                print(f"\nCase {idx}: {label}")
                # First three cases should be auto-merged (no prompt)
                if idx <= 3:
                    choice, merged = merge_record_pair(o_row, s_row, auto_choice='1')
                    print(f"  auto choice={choice}")
                    for k, v in merged.items():
                        if k in ('uuid',):
                            continue
                        print(f"   {k}: {v}")
                else:
                    # Fourth case: prompt the user when running interactively (like merge_output)
                    if sys.stdin.isatty():
                        print("Interactive run detected — prompting for merge choice for this case.")
                        choice, merged = merge_record_pair(o_row, s_row, auto_choice=None)
                        print(f"  choice={choice}")
                        if merged:
                            for k, v in merged.items():
                                if k in ('uuid',):
                                    continue
                                print(f"   {k}: {v}")
                    else:
                        # Non-interactive automated run: do not prompt, show summary and instructions
                        print("Non-interactive run: skipping prompt for case 4.")
                        print("Discrepancies for case 4:")
                        for col, d in merge_record_pair(o_row, s_row, auto_choice='0')[1].items():
                            if col in ('uuid',):
                                continue
                            print(f"   {col}: {d}")
                        print("To be prompted for this case run interactively: python -m data_ordering.merger_manual")