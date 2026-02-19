# Data Ordering Checker

A verification and metrics tool for analyzing the output of the `data_ordering` pipeline.

## Features

- **File Count Verification**: Counts files by type (non-duplicate images, duplicate images, text files, other files)
  - Analyzes actual file system structure
  - Cross-validates against Excel registries
  - Detects discrepancies

- **Taxonomy Analysis**: Extracts and displays all unique taxonomic information:
  - Macroclasses (e.g., Arthropoda, Pisces, Mollusca)
  - Classes (e.g., Insecta, Osteichthyes, Cephalopoda)
  - Determinations/Genera (detailed taxonomic determinations)

- **Distribution Metrics**: Provides insights into data organization:
  - Files per collection (Las Hoyas, Buenache, Montsec, etc.)
  - Files per macroclass
  - Files per class
  - Macroclass-Class distribution matrix

- **Export Capabilities**: Multiple output formats:
  - Console report (formatted text)
  - JSON (detailed metrics)
  - CSV (file counts and taxonomy)

## Installation

The checker is part of the main project. Simply use it from the project root:

```bash
cd /path/to/phdAlex_data_management
python -m data_ordering_checker.cli --output ./organized_data
```

## Usage

### Quick Report (Console)

Display a verification report in the console:

```bash
python -m data_ordering_checker.cli --output ./path/to/organized_data
```

Output includes:
- File system structure counts
- Excel registry counts
- Discrepancies (if any)
- Taxonomy information summary

### Export to JSON

Export the complete report with all metrics to a JSON file:

```bash
python -m data_ordering_checker.cli --output ./path/to/organized_data --json report.json
```

### Export to CSV

Export file counts and taxonomy information to CSV files:

```bash
python -m data_ordering_checker.cli --output ./path/to/organized_data --csv ./reports
```

This creates:
- `file_counts.csv` - File count comparison (structure vs registry)
- `taxonomy.csv` - All macroclasses, classes, and determinations

### Show All Metrics

Display comprehensive distribution metrics:

```bash
python -m data_ordering_checker.cli --output ./path/to/organized_data --all-metrics
```

Includes:
- Files per collection
- Files per macroclass
- Files per class (top 15)
- Duplicate files analysis

### Suppress Console Output

Use `--quiet` to suppress console output (useful when only exporting):

```bash
python -m data_ordering_checker.cli --output ./organized_data --json report.json --quiet
```

### Combine Multiple Exports

Generate all outputs at once:

```bash
python -m data_ordering_checker.cli \
  --output ./organized_data \
  --json report.json \
  --csv ./csv_reports \
  --all-metrics
```

## Python API

Use the checker programmatically in your own scripts:

```python
from pathlib import Path
from data_ordering_checker import DataOrderingChecker

# Initialize checker
checker = DataOrderingChecker(Path("./organized_data"))

# Get file counts from file system
file_counts = checker.count_files_in_structure()
print(f"Non-duplicate images: {file_counts.non_duplicate_images}")
print(f"Duplicate images: {file_counts.duplicate_images}")

# Get file counts from registries
registry_counts = checker.count_files_in_registries()

# Get taxonomy information
taxonomy = checker.get_taxonomy_info()
print(f"Macroclasses: {taxonomy.macroclasses}")
print(f"Classes: {taxonomy.classes}")
print(f"Determinations: {taxonomy.determinations}")

# Get detailed report
report = checker.get_detailed_report()
print(report)

# Export reports
checker.export_report_to_json(Path("report.json"))
checker.export_report_to_csv(Path("./reports"))

# Print formatted report
checker.print_report()
```

## Report Format

### Console Report

```
======================================================================
DATA ORDERING CHECKER - VERIFICATION REPORT
======================================================================

📁 FILE SYSTEM STRUCTURE COUNTS:
--------------------------------------------------
  Non-duplicate images: 1234
  Duplicate images:     45
  Text files:           67
  Other files:          8
  ─────────────────────────────────────────────
  TOTAL:                1354

📊 EXCEL REGISTRIES COUNTS:
--------------------------------------------------
  Non-duplicate images: 1234
  Duplicate images:     45
  Text files:           67
  Other files:          8
  ─────────────────────────────────────────────
  TOTAL:                1354

⚠️  DISCREPANCIES (Structure - Registry):
--------------------------------------------------
  ✓ No discrepancies found!

🔬 TAXONOMY INFORMATION:
--------------------------------------------------

  Macroclasses (5):
    • Arthropoda
    • Mollusca
    • Pisces
    ... and 2 more

  Classes (12):
    • Insecta
    • Cephalopoda
    ... and 10 more

  Determinations/Genera (89):
    • Genus 1
    • Genus 2
    ... and 87 more

📈 SUMMARY:
--------------------------------------------------
  Total files (structure):  1354
  Total files (registry):   1354
  Unique macroclasses:      5
  Unique classes:           12
  Unique determinations:    89
======================================================================
```

## What It Checks

1. **File Type Counting**
   - **Non-duplicate images**: Images in Las_Hoyas, Otras_Colecciones, and Revision_Manual folders
   - **Duplicate images**: Actual image files in Duplicados folder (not including registry file)
   - **Text files**: .txt, .csv, .md, .json files in Archivos_Texto folder
   - **Other files**: Any other file types in Otros_Archivos folder

2. **Registry Validation**
   - **anotaciones.xlsx**: Contains non-duplicate images (main registry)
   - **archivos_texto.xlsx**: Contains text files registry
   - **archivos_otros.xlsx**: Contains other files registry
   - **hashes.xlsx**: Contains MD5 and perceptual hashes for all images
   - **duplicados_registro.xlsx**: Separate registry for duplicate images (in Duplicados folder)
   - Compares counts from all registries to file system

3. **Taxonomy Extraction**
   - Reads `macroclass_label` from anotaciones.xlsx and duplicados_registro.xlsx
   - Reads `class_label` from both registries
   - Reads `genera_label` from both registries
   - Deduplicates and sorts results

4. **Discrepancy Detection**
   - Identifies differences between file system and registries
   - Checks if files exist on disk but not in registry
   - Checks if registry records exist but files are missing
   - Useful for catching data_ordering errors or manual interventions

## Output Directory Structure

The checker expects this structure created by `data_ordering`:

```
organized_data/
├── Las_Hoyas/
│   ├── Arthropoda/
│   │   ├── Insecta/
│   │   │   └── *.jpg
│   │   └── ...
│   └── ...
├── Otras_Colecciones/
│   ├── Buenache/
│   │   ├── Arthropoda/
│   │   │   ├── Insecta/
│   │   │   │   └── *.jpg
│   │   │   └── ...
│   │   └── ...
│   └── Montsec/
│       └── ...
├── Duplicados/
│   ├── (duplicate image files organized by macroclass/class)
│   └── duplicados_registro.xlsx (separate registry for duplicates)
├── Archivos_Texto/
│   └── (text files)
├── Otros_Archivos/
│   └── (other file types)
├── Revision_Manual/
│   └── (images flagged for manual review)
└── registries/
    ├── anotaciones.xlsx (non-duplicate images)
    ├── archivos_texto.xlsx (text files)
    ├── archivos_otros.xlsx (other files)
    └── hashes.xlsx (image hashes for deduplication)
```

## Exit Codes

- `0`: Success
- `1`: Error (invalid directory, missing registries, etc.)

## Troubleshooting

### "Output directory does not exist"
Ensure the path points to the root data_ordering output folder (the one containing Las_Hoyas, Otras_Colecciones, Duplicados, registries, etc.)

#### "Registries directory not found"
Ensure data_ordering has completed and created the `registries` subfolder. Check `pipeline_state.json` to verify completion.

#### File Counts Mismatch
Common causes:
1. **Files manually added/removed** after data_ordering completed
2. **data_ordering still running** - check `pipeline_state.json` for current stage
3. **Excel registry not yet written** - ensure pipeline reached REGISTRY_GENERATION stage
4. **Corrupted registry files** - check that `.xlsx` files are valid

#### Duplicates Count is 0 but files exist in Duplicados folder
- Check if `duplicados_registro.xlsx` exists in the Duplicados folder
- If missing, data_ordering may not have completed the REGISTRY_GENERATION stage
- Verify that duplicate files are actually in the Duplicados directory structure

#### "The checker is counting files that shouldn't exist"
The checker counts ALL files in certain directories. If data_ordering was run multiple times or files were manually added:
- Clean up extra files manually
- Re-run data_ordering with `--dry-run` first to preview changes

## Contributing

To add new metrics or validation checks:

1. Add methods to `DataOrderingChecker` or `MetricsCollector` class
2. Update `get_detailed_report()` if adding to main report
3. Update CLI help text if adding new options
4. Update this README with new functionality

## License

Same as parent project.
