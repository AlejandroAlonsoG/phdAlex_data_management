# Quick Reference - Data Ordering Checker

## Basic Usage

```bash
# Show report in console
python -m data_ordering_checker.cli --output ./organized_data

# Export to JSON
python -m data_ordering_checker.cli --output ./organized_data --json report.json

# Export to CSV
python -m data_ordering_checker.cli --output ./organized_data --csv ./reports

# Show everything with distributions
python -m data_ordering_checker.cli --output ./organized_data --all-metrics

# Combine all exports
python -m data_ordering_checker.cli --output ./organized_data \
  --json report.json --csv ./reports --all-metrics
```

## What Gets Counted

### ✅ Correct Counting (After Fixes)

**Non-Duplicate Images:**
- Files in: `Las_Hoyas/`
- Files in: `Otras_Colecciones/`
- Files in: `Revision_Manual/`
- Registry: `anotaciones.xlsx` row count
- Should be EQUAL ✓

**Duplicate Images:**
- Files in: `Duplicados/` (excluding .xlsx file)
- Registry: `Duplicados/duplicados_registro.xlsx` row count
- Should be EQUAL ✓

**Text Files:**
- Files in: `Archivos_Texto/`
- Registry: `archivos_texto.xlsx` row count
- Should be EQUAL ✓

**Other Files:**
- Files in: `Otros_Archivos/`
- Registry: `archivos_otros.xlsx` row count
- Should be EQUAL ✓

## Expected Output Example (60 + 6 files)

```
======================================================================
DATA ORDERING CHECKER - VERIFICATION REPORT
======================================================================

📁 FILE SYSTEM STRUCTURE COUNTS:
--------------------------------------------------
  Non-duplicate images: 60
  Duplicate images:     6
  Text files:           0
  Other files:          0
  ─────────────────────────────────────────────
  TOTAL:                66

📊 EXCEL REGISTRIES COUNTS:
--------------------------------------------------
  Non-duplicate images: 60
  Duplicate images:     6
  Text files:           0
  Other files:          0
  ─────────────────────────────────────────────
  TOTAL:                66

⚠️  DISCREPANCIES (Structure - Registry):
--------------------------------------------------
  ✓ No discrepancies found!

🔬 TAXONOMY INFORMATION:
--------------------------------------------------

  Macroclasses (X):
    • Arthropoda
    • Mollusca
    ...

  Classes (Y):
    • Insecta
    • Cephalopoda
    ...

  Determinations/Genera (Z):
    • Genus 1
    • Genus 2
    ...

📈 SUMMARY:
--------------------------------------------------
  Total files (structure):  66
  Total files (registry):   66
  Unique macroclasses:      X
  Unique classes:           Y
  Unique determinations:    Z
======================================================================
```

## Key Directories

| Path | Contains | Count From |
|------|----------|-----------|
| `Las_Hoyas/` | Non-duplicate images | File system |
| `Otras_Colecciones/` | Non-duplicate images | File system |
| `Duplicados/` | Duplicate images | File system |
| `Archivos_Texto/` | Text files | File system |
| `Otros_Archivos/` | Other files | File system |
| `registries/anotaciones.xlsx` | Non-duplicate metadata | Registry |
| `Duplicados/duplicados_registro.xlsx` | Duplicate metadata | Registry |
| `registries/archivos_texto.xlsx` | Text file metadata | Registry |
| `registries/archivos_otros.xlsx` | Other file metadata | Registry |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Counts mismatch | Registries not created yet | Verify pipeline completed REGISTRY_GENERATION stage |
| Duplicates count = 0 | Registry file missing | Check `Duplicados/duplicados_registro.xlsx` exists |
| Discrepancies found | Files added/removed manually | Clean up extra files or verify pipeline completed |
| "Registries not found" | Wrong output directory | Ensure path contains `registries/` subfolder |

## Python API Example

```python
from pathlib import Path
from data_ordering_checker import DataOrderingChecker

# Load checker
checker = DataOrderingChecker(Path("./organized_data"))

# Get counts
file_counts = checker.count_files_in_structure()
registry_counts = checker.count_files_in_registries()

print(f"Files: {file_counts.non_duplicate_images} non-dup, "
      f"{file_counts.duplicate_images} dup")
print(f"Registry: {registry_counts.non_duplicate_images} non-dup, "
      f"{registry_counts.duplicate_images} dup")

# Get taxonomy
taxonomy = checker.get_taxonomy_info()
print(f"Macroclasses: {taxonomy.to_dict()['macroclasses']}")

# Generate report
checker.print_report()

# Export
checker.export_report_to_json(Path("report.json"))
checker.export_report_to_csv(Path("./csv_output"))
```

## Files in This Package

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `__main__.py` | Module entry point |
| `checker.py` | Main verification class |
| `metrics.py` | Distribution metrics |
| `cli.py` | Command-line interface |
| `README.md` | User documentation |
| `OUTPUT_STRUCTURE.md` | Complete directory/registry reference |
| `FIXES_APPLIED.md` | Details of bug fixes |
| `COMPLETE_REFERENCE.md` | Exhaustive reference |
| `QUICK_REFERENCE.md` | This file |

## What Changed (vs Original)

✅ **Correct directory names**: `Duplicados/` not `Duplicates/`
✅ **Separate duplicate registry**: Reads `Duplicados/duplicados_registro.xlsx`
✅ **Accurate file counts**: No longer mixing non-dup/dup in main registry
✅ **Complete taxonomy**: Extracts from both registries
✅ **Better distribution metrics**: Aware of Otras_Colecciones structure
✅ **Proper duplicate counting**: Excludes .xlsx file from count
✅ **Revision folder included**: Counts `Revision_Manual/` as non-duplicates

## Notes

- All discrepancies should be 0 if pipeline completed successfully
- If discrepancies > 0, files may have been manually added/removed
- Duplicate files are PHYSICALLY SEPARATE from non-duplicates
- Taxonomy is extracted from BOTH registries for completeness
- Report format uses emoji for visual clarity in console
- Exports are idempotent (safe to run multiple times)
